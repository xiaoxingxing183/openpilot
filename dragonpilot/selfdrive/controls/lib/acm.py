import time
import numpy as np
from cereal import log
from openpilot.common.swaglog import cloudlog

# 匯入 Openpilot 原廠 MPC (模型預測控制) 相關的安全距離與參數計算公式
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, STOP_DISTANCE, get_safe_obstacle_distance,
  get_stopped_equivalence_factor, get_T_FOLLOW
)

# =========================================================
# 參數設定區
# =========================================================

# --- 滑行速度容許範圍 ---
SPEED_OFFSET_MIN_KPH = 0.0             
SPEED_OFFSET_MAX_FLAT_KPH = 15.0       
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0    

# --- 坡度訊號與判斷門檻 ---
PITCH_SMOOTH_ALPHA = 0.10              
PITCH_UPHILL_THRESHOLD = 0.050         
PITCH_DOWNHILL_THRESHOLD = -0.030      

# --- 坡度判斷門檻 (Soft Hold 動力保留專用) ---
SOFT_HOLD_PITCH_START = 0.050          
SOFT_HOLD_PITCH_MAX = 0.080            

# --- TTC 與 緊急狀況解除設定 ---
TTC_BP = [10., 30.]                    
TTC_V  = [3.0, 3.0]                    
EMERGENCY_TTC = 2.0                    
EMERGENCY_RELATIVE_SPEED = 10.0        
EMERGENCY_DECEL_THRESHOLD = -1.5       

# --- 系統冷卻與安全距離設定 ---
LEAD_COOLDOWN_TIME = 0.5               
SPEED_BP = [0., 10., 20., 30.]         
MIN_DIST_V = [5., 10., 15., 20.]       

# --- Soft Hold (柔和跟車/滑行介入) 設定 ---
SOFT_HOLD_RANGE_MIN = 0.70             
SOFT_HOLD_RANGE_MAX = 0.99             
SOFT_HOLD_TTC_THRESHOLD = 2.5          
VREL_DEBOUNCE_TIME = 0.6               

# 車速 (km/h) 對應 最高加速度限制 (m/s²) 的插值陣列
SOFT_HOLD_SPEED_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SOFT_HOLD_ACCEL_V  = [1.0,  0.80,  0.60,  0.40,  0.20,  0.0]


# =========================================================
# 邏輯模組 1：純滑行控制器 (專注處理無車滑行)
# =========================================================
class CoastingLogic:
  def __init__(self):
    self.active = False
    self.current_max_offset = 0.0
    self._has_lead = False
    self._last_lead_time = 0.0
    self._active_prev = False

  def check_emergency(self, lead, v_ego, current_time):
    if not lead or not lead.status:
      return False
    closing_speed = max(v_ego - lead.vLead, 0.1)
    lead_ttc = lead.dRel / closing_speed 
    relative_speed = v_ego - lead.vLead         
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    if (lead_ttc < EMERGENCY_TTC) or \
       (relative_speed > EMERGENCY_RELATIVE_SPEED) or \
       (lead.dRel < min_dist_for_speed and relative_speed > 0):
      self._last_lead_time = current_time
      if self.active:
        cloudlog.warning(f"ACM emergency disable: dRel={lead.dRel:.1f}m, TTC={lead_ttc:.1f}s, RelSpeed={relative_speed:.1f}m/s")
      return True
    return False

  def update_lead_status(self, lead, v_ego, current_time):
    if lead and lead.status:
      closing_speed = max(v_ego - lead.vLead, 0.1)
      lead_ttc = lead.dRel / closing_speed
      current_ttc_threshold = np.interp(v_ego, TTC_BP, TTC_V) 
      if lead_ttc < current_ttc_threshold:
        self._has_lead = True
        self._last_lead_time = current_time
      else:
        self._has_lead = False
    else:
      self._has_lead = False

  def update_states(self, enabled, user_ctrl_lon, v_ego, v_cruise, current_pitch, dtsc_is_active, current_time):
    if not enabled:
      self.active = False
      return

    # 判斷坡度帶來的超速容忍度
    if current_pitch < PITCH_DOWNHILL_THRESHOLD:
        self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH
    else:
        self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH

    upper_bound = v_cruise + (self.current_max_offset / 3.6)
    is_in_coast_window = (v_ego >= v_cruise and v_ego < upper_bound)
    in_cooldown = (current_time - self._last_lead_time) < LEAD_COOLDOWN_TIME

    # 綜合啟動條件
    should_activate = (not dtsc_is_active and
                       current_pitch <= PITCH_UPHILL_THRESHOLD and
                       not user_ctrl_lon and     
                       not self._has_lead and    
                       not in_cooldown and       
                       is_in_coast_window)
    
    self.active = should_activate

    # Log 紀錄
    just_disabled = self._active_prev and not self.active
    if self.active and not self._active_prev:
      pitch_deg = current_pitch * 57.2958
      cloudlog.info(f"ACM Coasting ON: v={v_ego*3.6:.0f}, pitch={pitch_deg:.1f}deg, Max+{self.current_max_offset:.0f}kph")
    elif just_disabled:
      cloudlog.info("ACM Coasting OFF")

    self._active_prev = self.active

  def process_trajectory(self, a_desired_trajectory, lead):
    traj = np.copy(a_desired_trajectory)
    if self.active:
      min_accel = np.min(traj)
      if min_accel < EMERGENCY_DECEL_THRESHOLD:
        cloudlog.warning(f"ACM aborting: MPC requested {min_accel:.2f} m/s² braking")
        self.active = False
      else:
        # 前方無車時，抹平微小的煞車意圖 (-0.3 ~ 0 m/s²)
        if not (lead is not None and lead.status):
          for i in range(len(traj)):
            if -0.3 < traj[i] < 0:
              traj[i] = 0.0
    return traj


# =========================================================
# 邏輯模組 2：柔和跟車控制器 (專注處理有車互動)
# =========================================================
class SoftHoldLogic:
  def __init__(self):
    # _soft_hold_factor: 動力保留係數，1.0 代表 100% 交給原廠 MPC，0.0 代表完全壓制動力 (輸出 current_soft_hold_accel)
    self._soft_hold_factor = 1.0
    
    # 防插隊/暴衝的計時器狀態
    self._vrel_high_start_time = 0.0      # 記錄前車開始加速遠離的初始時間
    self._vrel_high_active = False        # 標記目前是否處於「疑似前車遠離」的狀態

  def process_trajectory(self, a_desired_trajectory, v_ego, lead, current_pitch, t_follow):
    should_cancel_soft_hold = False
    current_time = time.monotonic()
    
    # 取得原廠 MPC 目前規劃出的最大加速度意圖
    mpc_max_accel_intent = np.max(a_desired_trajectory)
    
    # 檢查是否有有效的前車 (狀態存在且距離小於 100 公尺)
    has_valid_lead = lead is not None and lead.status and lead.dRel <= 100.0

    # ==========================================
    # 階段 1：評估是否需要強制解除 Soft Hold (退回原廠控制)
    # ==========================================
    if not has_valid_lead:
        # 情況 A：沒有前車，或是前車太遠 -> 解除限制，重置防插隊計時器
        should_cancel_soft_hold = True
        self._vrel_high_active = False
    else:
        # 情況 B：防插隊 Debounce 邏輯 (濾除市區機車鑽車縫造成的瞬間相對速度突波)
        if lead.vRel > 1.0: # 如果前車比我們快 1.0 m/s 以上 (看似正在駛離)
            if not self._vrel_high_active:
                # 剛偵測到駛離，開始計時，先不解除限制
                self._vrel_high_active = True
                self._vrel_high_start_time = current_time
            elif (current_time - self._vrel_high_start_time) > VREL_DEBOUNCE_TIME:
                # 確實駛離超過設定時間 (例如 0.6 秒) -> 安全解除限制，讓車子跟上
                should_cancel_soft_hold = True
        else:
            # 速度差沒那麼大，重置計時器狀態
            self._vrel_high_active = False
            
        # 情況 C：大陡坡防後滑 -> 超過 8% 坡度，強制把動力全權交給原廠 MPC 處理
        if current_pitch > SOFT_HOLD_PITCH_MAX:
            should_cancel_soft_hold = True
            
        # 情況 D：系統強烈需要動力 -> 防國道緩坡掉速，若 MPC 已經判定需要 > 0.4 m/s² 的加速，則不壓制它
        elif mpc_max_accel_intent > 0.4:
            should_cancel_soft_hold = True

    # ==========================================
    # 階段 2：計算跟車安全距離與目標壓制係數
    # ==========================================
    target_factor = 1.0   
    ratio = 10.0  
    v_ego_kph = v_ego * 3.6
    
    # 依據當前車速，查表得出 Soft Hold 啟動時允許的「最高基礎加速度」(例如 10km/h 以下允許 0.8 m/s²)
    current_soft_hold_accel = np.interp(v_ego_kph, SOFT_HOLD_SPEED_BP, SOFT_HOLD_ACCEL_V)
    
    is_lead_braking_strict = False

    if not should_cancel_soft_hold:
        is_lead_stopped = lead.vLead < 1.0  
        
        # 判斷前車是否處於「急煞或靜止」狀態 (車速越快，對急煞的判定門檻越嚴格)
        if v_ego_kph <= 10.0:
            is_lead_braking_strict = lead.aLeadK < -0.1 or is_lead_stopped
        elif v_ego_kph <= 30.0:
            is_lead_braking_strict = lead.aLeadK < -0.5 or is_lead_stopped
        elif v_ego_kph <= 40.0:
            is_lead_braking_strict = lead.aLeadK < -1.0 or is_lead_stopped
        else: 
            is_lead_braking_strict = lead.aLeadK < -1.25 or is_lead_stopped

        # 計算 TTC (碰撞時間) 與 距離比例 (實際距離 / 理想安全距離)
        closing_speed = max(v_ego - lead.vLead, 0.1)
        current_ttc = lead.dRel / closing_speed
        desired_dist = get_safe_obstacle_distance(v_ego, t_follow)
        lead_obstacle_dist = lead.dRel + get_stopped_equivalence_factor(lead.vLead)

        ratio = 10.0 if desired_dist < 0.1 else (lead_obstacle_dist / desired_dist)
        
        # 如果實際距離比理想安全距離多出 20% 以上，代表空間非常充足，解除動力壓制
        if ratio > 1.2:
            should_cancel_soft_hold = True

    # ==========================================
    # 階段 3：平滑過渡與計算最終輸出係數
    # ==========================================
    if should_cancel_soft_hold:
        # 解除壓制，目標係數回歸 1.0，使用較快的平滑速率 (0.40) 迅速恢復動力
        target_factor = 1.0
        alpha = 0.40  
    else:
        distance_factor = 1.0 
        
        # 只要不是大陡坡，且距離比例落入 70%~99% 區間、TTC < 2.5 秒，則距離係數降為 0.0 (準備壓制動力)
        if current_pitch <= SOFT_HOLD_PITCH_MAX:
            if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and current_ttc <= SOFT_HOLD_TTC_THRESHOLD:
                distance_factor = 0.0

        # 考慮兩車相對速度的係數
        v_rel_factor = np.interp(lead.vRel, [-2.0, 0.5], [0.0, 1.0])
        
        # 目標壓制係數取兩者最大值 (保證足夠的安全餘裕)
        target_factor = max(distance_factor, v_rel_factor)

        # 針對前車急煞/靜止，以及微上坡的特殊處理
        if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and is_lead_braking_strict:
            if current_pitch > SOFT_HOLD_PITCH_START:
                # 坡度在 5%~8% 之間，將目標係數從 0.0 平滑拉升到 1.0，防止微上坡起步往後溜
                smooth_factor = float(np.interp(current_pitch, [SOFT_HOLD_PITCH_START, SOFT_HOLD_PITCH_MAX], [0.0, 1.0]))
                target_factor = smooth_factor  
                current_soft_hold_accel = current_soft_hold_accel * smooth_factor 
            else:
                # 平地或下坡，直接給 0.0 徹底壓制多餘加速，準備煞停
                current_soft_hold_accel = 0.0
                target_factor = 0.0 

        # 決定平滑度 (放開壓制時較慢 alpha=0.10，加重壓制時較快 alpha=0.20)
        alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20 

    # 套用指數移動平均 (EMA) 讓係數變化滑順，避免頓挫
    self._soft_hold_factor = (1.0 - alpha) * self._soft_hold_factor + alpha * target_factor

    # ==========================================
    # 階段 4：攔截並修改 MPC 規劃的加速度軌跡
    # ==========================================
    traj = np.copy(a_desired_trajectory)
    
    # 只要係數不到 0.99 (代表需要介入壓制)
    if self._soft_hold_factor < 0.99:
        # 計算動態天花板：原廠加速意圖 * 壓制比例 + 基礎微量加速 * 壓制剩餘比例
        # np.maximum(traj, 0.0) 確保我們不會把原廠的煞車 (負值) 減弱，只壓制加速 (正值)
        dynamic_limit = np.maximum(traj, 0.0) * self._soft_hold_factor + current_soft_hold_accel * (1.0 - self._soft_hold_factor)
        
        # 取原廠軌跡和動態天花板的最小值，成功砍掉多餘的暴衝意圖
        traj = np.minimum(traj, dynamic_limit)

    return traj

# =========================================================
# 統一對外接口 (Facade) - 讓外部檔案無需修改直接呼叫
# =========================================================
class ACM:
  def __init__(self):
    self.enabled = False                  
    self.current_pitch = 0.0              
    self._is_first_pitch = True           
    self.personality = log.LongitudinalPersonality.standard 
    self._dtsc_is_active = False          
    self._is_normal_mode = True

    # 實例化子系統
    self.coasting = CoastingLogic()
    self.soft_hold = SoftHoldLogic()

  @property
  def active(self):
    # 對外暴露 coasting 狀態，維持原廠 UI 顯示燈號正常
    return self.coasting.active

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, mode='acc', personality=log.LongitudinalPersonality.standard, dtsc_is_active=False):
    self.personality = personality
    self._dtsc_is_active = dtsc_is_active 
    self._is_normal_mode = (mode == 'acc')

    if not self.enabled or len(cc.orientationNED) != 3:
      self.coasting.active = False
      return

    # 共用的坡度平滑處理 (EMA 濾波)
    new_pitch = cc.orientationNED[1]
    if self._is_first_pitch:
        self.current_pitch = new_pitch
        self._is_first_pitch = False
    else:
        self.current_pitch = PITCH_SMOOTH_ALPHA * new_pitch + (1.0 - PITCH_SMOOTH_ALPHA) * self.current_pitch

    current_time = time.monotonic()
    lead = rs.leadOne

    # 委託 CoastingLogic 處理無車滑行狀態更新
    if self.coasting.check_emergency(lead, v_ego, current_time):
      self.coasting.active = False
      return

    self.coasting.update_lead_status(lead, v_ego, current_time)
    self.coasting.update_states(self.enabled, user_ctrl_lon, v_ego, v_cruise, self.current_pitch, dtsc_is_active, current_time)

  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None, t_follow=None):
    if self._dtsc_
