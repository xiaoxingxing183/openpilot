import time
import numpy as np
from cereal import log
from openpilot.common.swaglog import cloudlog

# 匯入 Openpilot 原廠 MPC (模型預測控制) 相關的安全距離與參數計算公式
# 這些函數用於輔助判斷跟車距離與煞車時機
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, STOP_DISTANCE, get_safe_obstacle_distance,
  get_stopped_equivalence_factor, get_T_FOLLOW
)

# =========================================================
# 參數設定區 (定義了系統運作的各項物理限制與門檻)
# =========================================================

# --- 滑行速度容許範圍 ---
SPEED_OFFSET_MIN_KPH = 1.0             # 允許滑行的最低速度偏差
SPEED_OFFSET_MAX_FLAT_KPH = 15.0       # 平地時，允許超過設定定速的最大滑行速度 (例如定速100，最高允許滑行到115才強制煞車)
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0    # 下坡時，允許超過設定定速的最大滑行速度 (下坡較危險，容忍度較小)

# --- 坡度訊號與判斷門檻 (使用指數移動平均 EMA 濾波) ---
PITCH_SMOOTH_ALPHA_UP = 0.30           # [修正] 上坡濾波係數：數值較大(0.3)，代表對上坡的反應快，防止爬坡時無力或後溜
PITCH_SMOOTH_ALPHA_DOWN = 0.05         # [修正] 下坡與坑洞濾波係數：數值較小(0.05)，反應慢，用於過濾路面跳動或假下坡的雜訊
PITCH_UPHILL_THRESHOLD = 0.050         # 判定為上坡的坡度門檻 (弧度，約2.8度)
PITCH_DOWNHILL_THRESHOLD = -0.030      # 判定為下坡的坡度門檻 (弧度，約-1.7度)

# --- 坡度判斷門檻 (Soft Hold 動力保留專用) ---
SOFT_HOLD_PITCH_START = 0.050          # Soft Hold 開始平滑退出的坡度起點
SOFT_HOLD_PITCH_MAX = 0.080            # 超過此陡坡門檻，Soft Hold 完全不介入 (交給原廠處理以確保動力)

# --- TTC (Time-To-Collision 碰撞時間) 與 緊急狀況解除設定 ---
TTC_BP = [10., 30.]                    # 車速插值斷點 (m/s)，分別對應 36km/h 和 108km/h
TTC_V  = [3.0, 3.0]                    # 對應上述車速的安全碰撞時間門檻 (秒)。低於此時間視為有車阻擋，退出滑行
EMERGENCY_TTC = 2.0                    # 緊急碰撞時間門檻 (秒)，低於 2 秒判定為危急
EMERGENCY_RELATIVE_SPEED = 10.0        # 緊急相對速度門檻 (m/s)。若本車比前車快 36km/h 以上，極其危險
EMERGENCY_DECEL_THRESHOLD = -1.5       # 緊急減速門檻 (m/s²)。如果原廠 MPC 要求大於 1.5m/s² 的煞車，強制退出滑行

# --- 系統冷卻與安全距離設定 ---
LEAD_COOLDOWN_TIME = 0.5               # 失去前車目標後的冷卻時間(秒)，避免雷達閃爍導致頻繁切換狀態
SPEED_BP = [0., 10., 20., 30.]         # 絕對車速斷點 (m/s)
MIN_DIST_V = [5., 10., 15., 20.]       # 不同車速下的強制最小安全距離 (公尺)

# --- Soft Hold (柔和跟車/滑行介入) 設定 ---
SOFT_HOLD_RANGE_MIN = 0.70             # 前車距離比例的下限 (實際距離 / 安全距離)
SOFT_HOLD_RANGE_MAX = 0.99             # 前車距離比例的上限，在此區間內啟動柔和跟車
SOFT_HOLD_TTC_THRESHOLD = 2.5          # Soft hold 啟用的碰撞時間門檻 (秒)
VREL_DEBOUNCE_TIME = 0.6               # 高相對速度 (前車快速遠離) 的防抖時間(秒)，避免誤判

# 車速 (km/h) 對應 最高加速度限制 (m/s²) 的插值陣列
# 【註：這就是導致你高速踩煞車的關鍵設定，高速時被設定為 0.0】
SOFT_HOLD_SPEED_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SOFT_HOLD_ACCEL_V  = [1.0,  0.80,  0.60,  0.40,  0.20,  0.0]


# =========================================================
# 邏輯模組 1：純滑行控制器 (專注處理無車滑行)
# =========================================================
class CoastingLogic:
  def __init__(self):
    self.active = False                # 滑行狀態是否啟動
    self.current_max_offset = 0.0      # 當前允許的最大超速滑行容差
    self._has_lead = False             # 前方是否有車
    self._last_lead_time = 0.0         # 最後一次看到前車的時間
    self._active_prev = False          # 上一幀的滑行狀態 (用於列印日誌切換點)

  def check_emergency(self, lead, v_ego, current_time):
    """檢查是否發生緊急情況，需要立刻取消滑行"""
    if not lead or not lead.status:
      return False
    
    # 計算接近速度 (本車速 - 前車速)
    closing_speed = max(v_ego - lead.vLead, 0.1)
    # 計算 TTC = 距離 / 接近速度
    lead_ttc = lead.dRel / closing_speed 
    relative_speed = v_ego - lead.vLead         
    # 根據當前車速計算最小安全距離
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    # 滿足以下任一條件即為緊急狀況：TTC極短、相對速度過大、距離極近且正在靠近
    if (lead_ttc < EMERGENCY_TTC) or \
       (relative_speed > EMERGENCY_RELATIVE_SPEED) or \
       (lead.dRel < min_dist_for_speed and relative_speed > 0):
      self._last_lead_time = current_time
      if self.active:
        cloudlog.warning(f"ACM emergency disable: dRel={lead.dRel:.1f}m, TTC={lead_ttc:.1f}s, RelSpeed={relative_speed:.1f}m/s")
      return True
    return False

  def update_lead_status(self, lead, v_ego, current_time):
    """更新前車狀態，利用 TTC 來動態判斷是否算作 '有車阻擋'"""
    if lead and lead.status:
      closing_speed = max(v_ego - lead.vLead, 0.1)
      lead_ttc = lead.dRel / closing_speed
      # 動態計算當前車速對應的 TTC 門檻
      current_ttc_threshold = np.interp(v_ego, TTC_BP, TTC_V) 
      
      if lead_ttc < current_ttc_threshold:
        self._has_lead = True
        self._last_lead_time = current_time
      else:
        self._has_lead = False # TTC夠長，視為無車，允許滑行
    else:
      self._has_lead = False

  def update_states(self, enabled, user_ctrl_lon, v_ego, v_cruise, current_pitch, dtsc_is_active, current_time):
    """決定當下是否應該進入純滑行模式 (Coasting)"""
    if not enabled:
      self.active = False
      return

    # 根據下坡或平地/上坡，決定允許的滑行最高速度上限
    if current_pitch < PITCH_DOWNHILL_THRESHOLD:
        self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH
    else:
        self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH

    # 判斷車速是否在允許滑行的區間內 (大於定速，小於定速+容差)
    upper_bound = v_cruise + (self.current_max_offset / 3.6) # kph 轉 m/s
    is_in_coast_window = (v_ego >= v_cruise and v_ego < upper_bound)
    
    # 檢查是否剛失去前車目標還在冷卻期
    in_cooldown = (current_time - self._last_lead_time) < LEAD_COOLDOWN_TIME

    # 【滑行啟動的嚴格條件】：非防滑介入、非陡坡、使用者沒踩油門煞車、無前車阻擋、不在冷卻期、且在滑行速度區間內
    should_activate = (not dtsc_is_active and
                       current_pitch <= PITCH_UPHILL_THRESHOLD and
                       not user_ctrl_lon and     
                       not self._has_lead and    
                       not in_cooldown and       
                       is_in_coast_window)
    
    self.active = should_activate

    # 日誌紀錄：狀態切換時列印
    just_disabled = self._active_prev and not self.active
    if self.active and not self._active_prev:
      pitch_deg = current_pitch * 57.2958 # 弧度轉角度
      cloudlog.info(f"ACM Coasting ON: v={v_ego*3.6:.0f}, pitch={pitch_deg:.1f}deg, Max+{self.current_max_offset:.0f}kph")
    elif just_disabled:
      cloudlog.info("ACM Coasting OFF")

    self._active_prev = self.active

  def process_trajectory(self, a_desired_trajectory, lead):
    """處理滑行軌跡：當允許滑行時，把原廠微小的煞車指令抹平歸零，達到滑行效果"""
    traj = np.copy(a_desired_trajectory)
    if self.active:
      min_accel = np.min(traj)
      # 如果原廠 MPC 判定需要大腳煞車 (低於 -1.5m/s²)，立刻放棄滑行交回控制權
      if min_accel < EMERGENCY_DECEL_THRESHOLD:
        cloudlog.warning(f"ACM aborting: MPC requested {min_accel:.2f} m/s² braking")
        self.active = False
      else:
        # 如果前方沒有確實的前車，則執行抹平動作
        if not (lead is not None and lead.status):
          for i in range(len(traj)):
            # 將微小的負加速度 (0 到 -0.3 m/s²) 視為引擎阻力，強制歸零以實現純滑行
            if -0.3 < traj[i] < 0:
              traj[i] = 0.0
    return traj


# =========================================================
# 邏輯模組 2：柔和跟車控制器 (專注處理有車互動)
# =========================================================
class SoftHoldLogic:
  def __init__(self):
    self._soft_hold_factor = 1.0          # 動力輸出係數 (1.0=100%原廠動力，0.0=切斷動力/踩煞車)
    self._vrel_high_start_time = 0.0      # 記錄前車快速遠離的開始時間
    self._vrel_high_active = False        # 標記前車是否正在快速遠離

  def process_trajectory(self, a_desired_trajectory, v_ego, lead, current_pitch, t_follow):
    """處理柔和跟車邏輯：市區蠕行防插隊，平滑起步與煞停"""
    should_cancel_soft_hold = False
    current_time = time.monotonic()
    mpc_max_accel_intent = np.max(a_desired_trajectory)
    
    # 判斷是否有效前車
    has_valid_lead = lead is not None and lead.status

    # 狀態機 1：判斷是否需要強制取消柔和跟車
    if not has_valid_lead:
        should_cancel_soft_hold = True
        self._vrel_high_active = False
    else:
        # vRel > 1.0 代表前車以大於 1m/s (3.6km/h) 的速度遠離我們
        if lead.vRel > 1.0:
            if not self._vrel_high_active:
                self._vrel_high_active = True
                self._vrel_high_start_time = current_time
            # 如果前車持續遠離超過 VREL_DEBOUNCE_TIME (0.6秒)，取消 Soft Hold 讓原廠加速跟上
            elif (current_time - self._vrel_high_start_time) > VREL_DEBOUNCE_TIME:
                should_cancel_soft_hold = True
        else:
            self._vrel_high_active = False
            
        # 遇到較陡的上坡，或者原廠 MPC 打算強烈加速 (>0.4 m/s²) 時，取消介入
        if current_pitch > SOFT_HOLD_PITCH_MAX:
            should_cancel_soft_hold = True
        elif mpc_max_accel_intent > 0.4:
            should_cancel_soft_hold = True

    target_factor = 1.0   # 目標動力係數 (預設為不介入)
    ratio = 10.0          # 距離比例
    v_ego_kph = v_ego * 3.6
    
    # 根據當下車速，查表得到最大允許加速度 (這就是高速被切為0的元凶)
    current_soft_hold_accel = np.interp(v_ego_kph, SOFT_HOLD_SPEED_BP, SOFT_HOLD_ACCEL_V)
    is_lead_braking_strict = False

    # 狀態機 2：如果沒有被強制取消，則精細計算是否要限制動力
    if not should_cancel_soft_hold:
        # [修正] 市區塞車防插隊蠕行邏輯：前車低於 3.6km/h，且沒有正在遠離我們 (>0.3 m/s) 才算真的停止
        is_lead_stopped = (lead.vLead < 1.0) and (lead.vRel <= 0.3)  

        # [修正] 放寬嚴格煞車判定：如果前車正在緩慢蠕行遠離 (vRel > 0.5)，就不視為嚴格煞車，允許車輛跟上
        # 依照不同車速區段，賦予不同的前車煞車判定門檻 (aLeadK 是前車加速度，負值代表煞車)
        if v_ego_kph <= 10.0:
            is_lead_braking_strict = (lead.aLeadK < -0.1 or is_lead_stopped) and (lead.vRel < 0.5)
        elif v_ego_kph <= 30.0:
            is_lead_braking_strict = (lead.aLeadK < -0.5 or is_lead_stopped) and (lead.vRel < 0.5)
        elif v_ego_kph <= 40.0:
            is_lead_braking_strict = lead.aLeadK < -1.0 or is_lead_stopped
        else: 
            is_lead_braking_strict = lead.aLeadK < -1.25 or is_lead_stopped

        # 計算 TTC 和 預期安全距離
        closing_speed = max(v_ego - lead.vLead, 0.1)
        current_ttc = lead.dRel / closing_speed
        desired_dist = get_safe_obstacle_distance(v_ego, t_follow)
        # 前車實際距離 + 前車停止等效因子
        lead_obstacle_dist = lead.dRel + get_stopped_equivalence_factor(lead.vLead)

        # 距離比例 ratio = 實際距離 / 期望安全距離
        ratio = 10.0 if desired_dist < 0.1 else (lead_obstacle_dist / desired_dist)
        # 如果實際距離比預期安全距離還大 20% (距離夠遠)，取消介入
        if ratio > 1.2:
            should_cancel_soft_hold = True

    # 根據前面收集到的資訊，結算 target_factor (目標動力係數)
    if should_cancel_soft_hold:
        target_factor = 1.0 # 恢復 100% 動力
        alpha = 0.40        # 恢復速度較快 (濾波係數大)
    else:
        distance_factor = 1.0 
        if current_pitch <= SOFT_HOLD_PITCH_MAX:
            # 如果跟車距離在目標範圍內，且 TTC 小於 2.5 秒，則將距離動力係數設為 0
            if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and current_ttc <= SOFT_HOLD_TTC_THRESHOLD:
                distance_factor = 0.0

        # 將相對速度映射到 0~1 的係數，前車靠近得越快(-2.0)，v_rel_factor 越趨近於 0
        v_rel_factor = np.interp(lead.vRel, [-2.0, 0.5], [0.0, 1.0])
        # 取距離和速度兩者中最保守(最大)的值
        target_factor = max(distance_factor, v_rel_factor)

        # 特別處理：如果前車正在嚴格煞車，且距離適中
        if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and is_lead_braking_strict:
            if current_pitch > SOFT_HOLD_PITCH_START:
                # 遇到微微上坡，根據坡度平滑保留部分動力，避免溜車
                smooth_factor = float(np.interp(current_pitch, [SOFT_HOLD_PITCH_START, SOFT_HOLD_PITCH_MAX], [0.0, 1.0]))
                target_factor = smooth_factor  
                current_soft_hold_accel = current_soft_hold_accel * smooth_factor 
            else:
                # 平地或下坡，直接切斷動力，準備煞車
                current_soft_hold_accel = 0.0
                target_factor = 0.0 

        # 進入柔和跟車的速度較慢(alpha=0.10)，退出恢復動力的速度較快(alpha=0.20)
        alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20 

    # 套用指數移動平均 (EMA) 使動力變化平滑，不突兀
    self._soft_hold_factor = (1.0 - alpha) * self._soft_hold_factor + alpha * target_factor

    # 最終套用到油門/煞車軌跡上
    traj = np.copy(a_desired_trajectory)
    # 只要 factor 小於 0.99，代表開始介入限制動力
    if self._soft_hold_factor < 0.99:
        # 原廠正加速度 * 限制係數 + 自定義最大允許加速度 * (1 - 限制係數)
        dynamic_limit = np.maximum(traj, 0.0) * self._soft_hold_factor + current_soft_hold_accel * (1.0 - self._soft_hold_factor)
        # 用 np.minimum 限制最終輸出的加速度
        traj = np.minimum(traj, dynamic_limit)

    return traj


# =========================================================
# 統一對外接口 (Facade) - 讓外部檔案無需修改直接呼叫
# =========================================================
class ACM:
  def __init__(self):
    self.enabled = False                  # 總開關
    self.current_pitch = 0.0              # 當前平滑化後的坡度
    self._is_first_pitch = True           # 是否為第一次接收坡度資料
    self.personality = log.LongitudinalPersonality.standard # 跟車性格 (遠/中/近)
    self._dtsc_is_active = False          # 循跡防滑系統是否作動中
    self._is_normal_mode = True           # 是否處於一般 ACC 模式

    self.coasting = CoastingLogic()       # 實例化純滑行控制器
    self.soft_hold = SoftHoldLogic()      # 實例化柔和跟車控制器

  @property
  def active(self):
    """提供對外查詢當下是否正在純滑行狀態"""
    return self.coasting.active

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, mode='acc', personality=log.LongitudinalPersonality.standard, dtsc_is_active=False):
    """每幀更新車輛與環境狀態"""
    self.personality = personality
    self._dtsc_is_active = dtsc_is_active 
    self._is_normal_mode = (mode == 'acc')

    # 如果 ACM 未開啟，或者車身姿態資料不完整，關閉滑行
    if not self.enabled or len(cc.orientationNED) != 3:
      self.coasting.active = False
      return

    # [修正] 非對稱坡度平滑處理 (Asymmetric EMA Filter)
    # 取出車輛俯仰角 (Pitch)，cc.orientationNED[1] 是從卡爾曼濾波器算出的坡度
    new_pitch = cc.orientationNED[1]
    if self._is_first_pitch:
        self.current_pitch = new_pitch
        self._is_first_pitch = False
    else:
        # 動態調整濾波係數：遇到上坡反應快，遇到坑洞/下坡濾除強
        alpha = PITCH_SMOOTH_ALPHA_UP if new_pitch > self.current_pitch else PITCH_SMOOTH_ALPHA_DOWN
        self.current_pitch = alpha * new_pitch + (1.0 - alpha) * self.current_pitch

    current_time = time.monotonic()
    lead = rs.leadOne # 獲取主要跟車目標

    # 如果純滑行模組判定遇到緊急狀況，立刻關閉滑行並退出
    if self.coasting.check_emergency(lead, v_ego, current_time):
      self.coasting.active = False
      return

    # 更新滑行模組的前車狀態，並計算是否進入滑行
    self.coasting.update_lead_status(lead, v_ego, current_time)
    self.coasting.update_states(self.enabled, user_ctrl_lon, v_ego, v_cruise, self.current_pitch, dtsc_is_active, current_time)

  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None, t_follow=None):
    """
    修改原廠規劃的加速度軌跡 (核心執行函數)
    會依次被 Coasting (抹平阻力) 和 SoftHold (限制動力) 處理
    """
    # 如果防滑系統作動，或不是標準 ACC 模式，原封不動回傳原廠軌跡，保證安全
    if self._dtsc_is_active or not self._is_normal_mode:
        return a_desired_trajectory

    # 獲取根據性格動態計算的跟車時距
    if t_follow is None:
        t_follow = get_T_FOLLOW(self.personality)

    # 步驟 1：通過純滑行控制器處理 (如果是純滑行狀態，會把微小的負加速度歸零)
    traj = self.coasting.process_trajectory(a_desired_trajectory, lead)
    # 步驟 2：通過柔和跟車控制器處理 (限制加速度，平滑加減速)
    traj = self.soft_hold.process_trajectory(traj, v_ego, lead, self.current_pitch, t_follow)
    
    return traj
