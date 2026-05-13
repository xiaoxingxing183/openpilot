import time
import numpy as np
from cereal import log
from openpilot.common.swaglog import cloudlog

# 匯入 Openpilot 原廠 MPC 相關公式與常數
# 確保我們計算距離的邏輯與原廠底層 100% 同步
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, STOP_DISTANCE, get_safe_obstacle_distance,
  get_stopped_equivalence_factor, get_T_FOLLOW
)

# =========================================================
# 參數設定區 (Configuration Parameters)
# =========================================================

# --- 加速意圖偵測參數 ---
TRAJECTORY_HORIZON  = 6      # 預判原廠規劃軌跡的前 6 個點 (約涵蓋 0.6 秒的未來)
INTENT_LOOKAHEAD    = 3      # 在這 6 個點中，若有 3 個點符合加速閾值，即判定車輛「有意圖」加速
INTENT_V_LOW        = 0.0    # 低速基準線 (0 km/h)
INTENT_V_HIGH       = 22.22  # 高速基準線 (約 80 km/h, 單位: m/s)
INTENT_FRAMES_LOW   = 0      # 低速時，需連續 0 幀 (即只要當下這 1 幀偵測到，立刻 0 延遲放行)
INTENT_FRAMES_HIGH  = 5     # 高速時，需要連續 5 幀偵測到加速才放行 (防止高速巡航時的震盪誤判)

# --- 巡航與坡度參數 ---
SPEED_OFFSET_MIN_KPH = 1.0             # 容許的最小超速滑行寬容值
SPEED_OFFSET_MAX_FLAT_KPH = 15.0       # 平路時，最大容許超速滑行 15 km/h
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0    # 下坡時，為了安全，最大容許超速縮減為 5 km/h

PITCH_SMOOTH_ALPHA_UP = 0.30           # 坡度變化平滑係數 (上坡反應較快)
PITCH_SMOOTH_ALPHA_DOWN = 0.05         # 坡度變化平滑係數 (下坡反應較慢，防雜訊)
PITCH_UPHILL_THRESHOLD = 0.050         # 判定為上坡的閾值 (5% 坡度)
PITCH_DOWNHILL_THRESHOLD = -0.030      # 判定為下坡的閾值 (-3% 坡度)

SOFT_HOLD_PITCH_START = 0.050          # 開始遞減 Soft Hold 煞車力道的坡度起點
SOFT_HOLD_PITCH_MAX = 0.080            # 完全取消 Soft Hold 的最大陡坡閾值

# --- 緊急避險與雷達參數 ---
TTC_BP = [10., 30.]                    # 碰撞時間 (TTC) 判定車速節點
TTC_V  = [3.0, 3.0]                    # 對應的 TTC 閾值 (秒)
EMERGENCY_TTC = 2.0                    # 極度危險 TTC 閾值 (小於 2 秒立即還原動力與煞車)
EMERGENCY_RELATIVE_SPEED = 10.0        # 極度危險相對速差 (前車比我慢 10 m/s 以上)
EMERGENCY_DECEL_THRESHOLD = -1.5       # 原廠規劃急煞的減速度閾值 (大於 1.5 m/s^2 視為緊急)

LEAD_COOLDOWN_TIME = 0.5               # 失去前車目標後的冷卻時間 (0.5秒內不貿然動作)
SPEED_BP = [0., 10., 20., 30.]         # 最小安全距離判定的車速節點
MIN_DIST_V = [5., 10., 15., 20.]       # 對應車速的絕對最小安全物理距離 (公尺)

# =========================================================
# 🚀 物理公式專用：滑行與防震盪閾值 (核心修改區)
# =========================================================
# 這裡的數值代表「實體距離佔動態目標距離的百分比」
SOFT_HOLD_RANGE_MAX = 0.90            # 進入純滑行的最高界線 (低於目標距離 90% 啟動滑行，斷開原廠煞車)
SOFT_HOLD_RANGE_MIN = 0.70             # 交還控制權的最低界線 (低於目標距離 70% 代表太近，交給原廠重煞)

RATIO_ENTER_THRESHOLD = 1.00           # 空間充裕界線：實體距離大於目標 100% 時，徹底解除滑行允許補油門
RATIO_EXIT_THRESHOLD = 0.98            # 重新進入滑行的判斷線

SOFT_HOLD_TTC_THRESHOLD = 2.5          # 允許 Soft Hold 介入的最大 TTC
VREL_DEBOUNCE_TIME = 0.6               # 高速差防震盪計時器

# 微煞車力道對照表 (車速 km/h 對應 加速度 m/s^2)
SOFT_HOLD_SPEED_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SOFT_HOLD_ACCEL_V  = [1.1,  0.90,  0.70,  0.50,  0.30,  0.10]

TARGET_FACTOR_FILTER_ALPHA = 0.3       # 最終輸出因子的平滑濾波係數
SOFT_HOLD_HYSTERESIS_TIME = 1.0        # 狀態切換的滯後防震盪時間 (秒)

# =========================================================
# 邏輯模組 1：純滑行控制器 (無車或極度安全時的空檔滑行)
# =========================================================
class CoastingLogic:
  def __init__(self):
    self.active = False                
    self.current_max_offset = 0.0      
    self._has_lead = False             
    self._last_lead_time = 0.0         
    self._active_prev = False          

  def check_emergency(self, lead, v_ego, current_time):
    """檢查是否處於緊急狀況 (例如快撞上、前車急煞)，若是則強制退出滑行"""
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
      return True
    return False

  def update_lead_status(self, lead, v_ego, current_time):
    """更新雷達前車狀態，過濾雜訊"""
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
    """綜合判斷是否允許開啟無車純滑行模式"""
    if not enabled:
      self.active = False
      return
      
    # 根據坡度決定容許的超速上限
    if current_pitch < PITCH_DOWNHILL_THRESHOLD:
        self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH
    else:
        self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH
        
    upper_bound = v_cruise + (self.current_max_offset / 3.6) 
    is_in_coast_window = (v_ego >= v_cruise and v_ego < upper_bound)
    in_cooldown = (current_time - self._last_lead_time) < LEAD_COOLDOWN_TIME
    
    # 啟動條件：無人工干預、無前車威脅、在容許超速範圍內、非陡坡
    should_activate = (not dtsc_is_active and
                       current_pitch <= PITCH_UPHILL_THRESHOLD and
                       not user_ctrl_lon and     
                       not self._has_lead and    
                       not in_cooldown and       
                       is_in_coast_window)
    self.active = should_activate
    self._active_prev = self.active

  def process_trajectory(self, a_desired_trajectory, lead):
    """處理軌跡，將微弱的煞車指令抹平以實現純滑行"""
    traj = np.copy(a_desired_trajectory)
    if self.active:
      min_accel = np.min(traj)
      if min_accel < EMERGENCY_DECEL_THRESHOLD:
        self.active = False # 原廠要急煞了，立刻交還控制權
      else:
        if not (lead is not None and lead.status):
          # 將 -0.15 到 0.0 之間的微弱煞車歸零，達成順暢滑行
          mask = (traj > -0.15) & (traj < 0.0)
          traj[mask] = traj[mask] * (np.abs(traj[mask]) / 0.15)
    return traj


# =========================================================
# 邏輯模組 2：物理級距柔和滑行控制器 (防震盪鎖定版)
# =========================================================
class SoftHoldLogic:
  def __init__(self):
    self._soft_hold_factor = 1.0          
    self._vrel_high_start_time = 0.0      
    self._vrel_high_active = False        
    
    self._last_lead_time = 0.0            
    self._last_target_factor = 1.0        
    self._last_soft_hold_accel = 0.0      

    # 新版意圖加速計數器 (移除所有強度與過濾器)
    self.accel_intent_counter = 0
    self.intent_accelerating = False
    
    self._ratio_hysteresis_state = False  
    self._target_factor_smooth = 1.0      
    
    self._last_stable_cancel_state = False
    self._state_change_time = 0.0
    
    # 核心狀態鎖：防止低速煞停時因為比例波動而「放開又煞車」
    self._is_soft_holding = False

  def process_trajectory(self, a_desired_trajectory, v_ego, lead, current_pitch, t_follow):
    should_cancel_soft_hold = False
    current_time = time.monotonic()
    
    # --- 1. 動態意圖偵測 (預判是否該補油門了) ---
    recent_trajectory = a_desired_trajectory[:TRAJECTORY_HORIZON]
    has_valid_lead = lead is not None and lead.status
    
    # 依據車速動態計算需要的觀察幀數 (低速 0 幀立刻反應，高速 20 幀防雜訊)
    v_ratio = max(0.0, min((v_ego - INTENT_V_LOW) / (INTENT_V_HIGH - INTENT_V_LOW), 1.0))
    dynamic_intent_frames = int(round(INTENT_FRAMES_LOW + v_ratio * (INTENT_FRAMES_HIGH - INTENT_FRAMES_LOW)))
    
    # 偵測原廠軌跡中是否包含明確的加速指令
    moment_accel = sum(1 for a in recent_trajectory if a > 0.05) >= INTENT_LOOKAHEAD and (lead.vRel > 0.05 if has_valid_lead else True)

    target_factor = 1.0   
    v_ego_kph = v_ego * 3.6
    current_soft_hold_accel = np.interp(v_ego_kph, SOFT_HOLD_SPEED_BP, SOFT_HOLD_ACCEL_V)
    is_lead_braking_strict = False
    skip_state_2 = False

    # --- 2. 狀態機與意圖直接放行 ---
    if not has_valid_lead:
        self._vrel_high_active = False
        # 失去前車時，計數器與狀態直接乾淨歸零
        self.accel_intent_counter = 0
        self.intent_accelerating = False

        # 短暫失去目標時維持上一幀狀態 (防雷達閃爍)
        if (current_time - self._last_lead_time) < 0.5:
            if self._last_soft_hold_accel >= 0.0:
                target_factor = self._last_target_factor
                current_soft_hold_accel = self._last_soft_hold_accel
            else:
                target_factor = 0.0
                current_soft_hold_accel = 0.0
            should_cancel_soft_hold = False
            skip_state_2 = True 
        else:
            should_cancel_soft_hold = True
            skip_state_2 = True
    else:
        self._last_lead_time = current_time 
        
        # 乾淨的連續計數器：只要有加速意圖就累加，一沒有就立刻斷開歸零
        if moment_accel:
            self.accel_intent_counter += 1
        else:
            self.accel_intent_counter = 0

        # 【條件達成即刻放行】
        # 只要 moment_accel 成立，且累積幀數大於等於當下車速規定的閾值，直接觸發放行
        if moment_accel and self.accel_intent_counter >= dynamic_intent_frames:
            self.intent_accelerating = True
            should_cancel_soft_hold = True
        else:
            self.intent_accelerating = False

        # 速差過大保護 (防高速逼近)
        if lead.vRel > 1.0:
            if not self._vrel_high_active:
                self._vrel_high_active = True
                self._vrel_high_start_time = current_time
            elif (current_time - self._vrel_high_start_time) > VREL_DEBOUNCE_TIME:
                should_cancel_soft_hold = True
        else:
            self._vrel_high_active = False
            
        if current_pitch > SOFT_HOLD_PITCH_MAX:
            should_cancel_soft_hold = True

    # --- 3. 核心物理距離計算與判定 ---
    if not skip_state_2:
        # 判斷前車是否處於煞車或靜止狀態，決定我們是否要疊加柔和煞車 (Soft Hold Accel)
        is_lead_stopped = (lead.vLead < 1.0) and (lead.vRel <= 0.3)
        if v_ego_kph <= 10.0:
            is_lead_braking_strict = (lead.aLeadK < -0.1 or is_lead_stopped) and (lead.vRel < 0.5)
        elif v_ego_kph <= 30.0:
            is_lead_braking_strict = (lead.aLeadK < -0.5 or is_lead_stopped) and (lead.vRel < 0.5)
        elif v_ego_kph <= 40.0:
            is_lead_braking_strict = lead.aLeadK < -1.0 or is_lead_stopped
        else: 
            is_lead_braking_strict = lead.aLeadK < -1.25 or is_lead_stopped

        closing_speed = max(v_ego - lead.vLead, 0.1)
        current_ttc = lead.dRel / closing_speed
        
        # ==========================================================
        # 🚀 真實動態目標距離計算 (MPC 物理同步)
        # ==========================================================
        safe_v_lead = max(0.0, lead.vLead)
        # 目標距離 = (自車完全煞停所需距離) - (前車動能預計滑行距離)
        mpc_target = get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(safe_v_lead)
        # 底線防護：目標距離絕對不能小於原廠預設的基礎物理煞停距離
        dynamic_target_dist = max(mpc_target, STOP_DISTANCE)
        # 物理比例：當下實體距離 佔 安全目標距離 的百分比
        ratio = lead.dRel / dynamic_target_dist
        
        if not should_cancel_soft_hold:
            if ratio > RATIO_ENTER_THRESHOLD:
                self._ratio_hysteresis_state = True
            elif ratio < RATIO_EXIT_THRESHOLD:
                self._ratio_hysteresis_state = False
            
            if self._ratio_hysteresis_state:
                should_cancel_soft_hold = True
        
        # ==========================================================
        # 🔴 終極防線：6 公尺絕對棄權原則 (最高優先級覆寫)
        # 只要實體距離逼近到原廠預設的極限煞停區 (STOP_DISTANCE)，
        # ACM 強制罷工，將最後的精準煞停權限 100% 交還給原廠 MPC 處理，防止邏輯打架錯亂。
        # ==========================================================
        if lead.dRel <= STOP_DISTANCE:
            should_cancel_soft_hold = True
            self._is_soft_holding = False
            self._ratio_hysteresis_state = True # 同步重置防震盪狀態

    # --- 4. 狀態機防震盪計時器 (Debounce) ---
    if should_cancel_soft_hold != self._last_stable_cancel_state:
        if self._state_change_time == 0.0:
            self._state_change_time = current_time
        elif (current_time - self._state_change_time) > (SOFT_HOLD_HYSTERESIS_TIME / 2):
            self._last_stable_cancel_state = should_cancel_soft_hold
            self._state_change_time = 0.0
    else:
        self._state_change_time = 0.0

    should_cancel_soft_hold = self._last_stable_cancel_state

    # --- 5. 最終動力與煞車因子的計算與覆寫 ---
    if should_cancel_soft_hold:
        target_factor = 1.0  # 1.0 代表 100% 聽從原廠 MPC 的軌跡 (還原動力/煞車)
        alpha = 0.60 if self.intent_accelerating else 0.30 
        self._is_soft_holding = False # 釋放狀態鎖定
        
    elif not skip_state_2: 
        distance_factor = 1.0 
        if current_pitch <= SOFT_HOLD_PITCH_MAX:
            # === 防震盪狀態鎖定邏輯 (State Lock) ===
            # 1. 若距離太近，解除鎖定，讓 MPC 接手重煞保命
            if ratio <= SOFT_HOLD_RANGE_MIN:
                self._is_soft_holding = False
            # 2. 若前車走遠，空間充裕，解除鎖定準備加速
            elif ratio >= RATIO_ENTER_THRESHOLD:
                self._is_soft_holding = False
            # 3. 進入安全的滑行口袋區間，啟動鎖定！
            # (一旦鎖定，即使低速時分母縮水導致 ratio 暴增，系統也會堅持踩住煞車，直到前車真的開走)
            elif ratio < SOFT_HOLD_RANGE_MAX and current_ttc <= SOFT_HOLD_TTC_THRESHOLD:
                self._is_soft_holding = True

            # 只要處於鎖定滑行狀態，強制切斷原廠動力 (factor = 0.0)
            if self._is_soft_holding:
                distance_factor = 0.0

        v_rel_factor = np.interp(lead.vRel, [-2.0, 0.5], [0.0, 1.0])
        target_factor = max(distance_factor, v_rel_factor)

        # 決定是否疊加模組自帶的柔和煞車力道 (Soft Hold Accel)
        if self._is_soft_holding and is_lead_braking_strict:
            if current_pitch > SOFT_HOLD_PITCH_START:
                # 坡度過大時，遞減柔和煞車力道，避免與重力衝突
                smooth_factor = float(np.interp(current_pitch, [SOFT_HOLD_PITCH_START, SOFT_HOLD_PITCH_MAX], [0.0, 1.0]))
                target_factor = smooth_factor  
                current_soft_hold_accel = current_soft_hold_accel * smooth_factor 
            else:
                if is_lead_stopped:
                    # 前車靜止時，依據車速給予微弱的煞車力道協助平順停下
                    current_soft_hold_accel = float(np.interp(v_ego_kph, [0.0, 150.0], [0.0, -0.30]))
                elif v_ego_kph >= 50.0:
                    # 高速時若前車急煞，按比例補償煞車力道
                    if lead.vRel < -0.1 and lead.aLeadK <= -1.5:
                        dynamic_brake = lead.aLeadK * 0.30
                        current_soft_hold_accel = np.clip(dynamic_brake, -1.0, 0.0)
                    else:
                        current_soft_hold_accel = 0.0 
                else:
                    current_soft_hold_accel = 0.0
                target_factor = 0.0 

        alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20 
    else:
        alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20 

    self._last_target_factor = target_factor
    self._last_soft_hold_accel = current_soft_hold_accel
    
    # 透過低通濾波器平滑因子，確保加減速過渡不頓挫
    self._soft_hold_factor = (1.0 - alpha) * self._soft_hold_factor + alpha * target_factor
    self._target_factor_smooth = (1.0 - TARGET_FACTOR_FILTER_ALPHA) * self._target_factor_smooth + TARGET_FACTOR_FILTER_ALPHA * self._soft_hold_factor

    # --- 6. 軌跡合成 (Trajectory Blending) ---
    traj = np.copy(a_desired_trajectory)
    if self._target_factor_smooth < 0.99: 
        hold_strength = 1.0 - self._target_factor_smooth
        # 將原廠的加速指令與我們的微煞車指令依權重融合
        dynamic_limit = np.maximum(traj, 0.0) * self._target_factor_smooth + current_soft_hold_accel * hold_strength
        blend_factor = 0.5
        exceeds_mask = traj > dynamic_limit
        traj = np.where(exceeds_mask, dynamic_limit * blend_factor + traj * (1.0 - blend_factor), traj)

    return traj


# =========================================================
# 統一對外接口 (主程式呼叫的類別)
# =========================================================
class ACM:
  def __init__(self):
    self.enabled = False                  
    self.current_pitch = 0.0              
    self._is_first_pitch = True           
    self.personality = log.LongitudinalPersonality.standard 
    self._dtsc_is_active = False          
    self._is_normal_mode = True           

    self.coasting = CoastingLogic()       
    self.soft_hold = SoftHoldLogic()      

  @property
  def active(self):
    return self.coasting.active

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, mode='acc', personality=log.LongitudinalPersonality.standard, dtsc_is_active=False):
    """每幀更新車輛與雷達狀態"""
    self.personality = personality
    self._dtsc_is_active = dtsc_is_active 
    self._is_normal_mode = (mode == 'acc')

    if not self.enabled or len(cc.orientationNED) != 3:
      self.coasting.active = False
      return

    # 更新並平滑化道路坡度資訊
    new_pitch = cc.orientationNED[1]
    if self._is_first_pitch:
        self.current_pitch = new_pitch
        self._is_first_pitch = False
    else:
        alpha = PITCH_SMOOTH_ALPHA_UP if new_pitch > self.current_pitch else PITCH_SMOOTH_ALPHA_DOWN
        self.current_pitch = alpha * new_pitch + (1.0 - alpha) * self.current_pitch

    current_time = time.monotonic()
    lead = rs.leadOne 

    if self.coasting.check_emergency(lead, v_ego, current_time):
      self.coasting.active = False
      return

    self.coasting.update_lead_status(lead, v_ego, current_time)
    self.coasting.update_states(self.enabled, user_ctrl_lon, v_ego, v_cruise, self.current_pitch, dtsc_is_active, current_time)

  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None):
    """注入並修改原廠輸出的縱向控制軌跡"""
    if self._dtsc_is_active or not self._is_normal_mode:
        return a_desired_trajectory

    # 取得原廠對應性格的預設跟車秒距
    t_follow = get_T_FOLLOW(self.personality)

    # 1. 先經過純滑行邏輯處理
    traj = self.coasting.process_trajectory(a_desired_trajectory, lead)
    # 2. 再經過柔和跟車滑行邏輯處理 (物理公式防震盪版)
    traj = self.soft_hold.process_trajectory(traj, v_ego, lead, self.current_pitch, t_follow)
    
    return traj
