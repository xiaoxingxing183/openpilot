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
# ACM (Active Coasting Management) 主動滑行管理系統 - 參數設定區
# =========================================================

# --- 滑行速度容許範圍 ---
SPEED_OFFSET_MIN_KPH = 1.0             # 巡航速限下方容許範圍 (km/h)
SPEED_OFFSET_MAX_FLAT_KPH = 15.0       # 平路巡航速限上方最高容許範圍 (km/h)
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0    # 下坡時巡航速限上方最高容許範圍 (km/h)

# --- 坡度判斷門檻 ---
PITCH_UPHILL_THRESHOLD = 0.015         # 上坡判定門檻 (弧度)
PITCH_DOWNHILL_THRESHOLD = -0.030      # 下坡判定門檻 (弧度)

# --- TTC (Time To Collision) 碰撞時間設定 ---
TTC_BP = [10., 30.]                    # 車速斷點 (m/s)
TTC_V  = [3.0, 3.0]                    # 對應車速的 TTC 啟動門檻 (秒)

# --- 緊急狀況解除滑行設定 ---
EMERGENCY_TTC = 2.0                    # 緊急 TTC 門檻，低於此值解除滑行
EMERGENCY_RELATIVE_SPEED = 10.0        # 緊急相對速差門檻 (m/s)
EMERGENCY_DECEL_THRESHOLD = -1.5       # MPC 請求減速度超過此值時強制中止 ACM

# --- 系統冷卻與安全距離設定 ---
LEAD_COOLDOWN_TIME = 0.5               # 失去前車後的冷卻時間 (秒)
SPEED_BP = [0., 10., 20., 30.]         # 車速斷點 (m/s)
MIN_DIST_V = [5., 10., 15., 20.]       # 對應車速的最小安全距離 (公尺)

# --- Soft Hold (柔和跟車/滑行介入) 設定 ---
SOFT_HOLD_RANGE_MAX = 0.99             # Soft Hold 介入的距離比例上限 (相對於安全距離)
SOFT_HOLD_TTC_THRESHOLD = 2.5          # TTC 小於此數值時，開始切斷動力進行滑行緩衝

# 車速 (km/h) 對應 最高加速度限制 (m/s²) 的插值陣列
# 目的：在接近前車時，限制車輛的補油門力度，避免衝刺感
SOFT_HOLD_SPEED_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SOFT_HOLD_ACCEL_V  = [1.0,  0.80,  0.60,  0.40,  0.20,  0.0]


class ACM:
  def __init__(self):
    self.enabled = False                  # ACM 總開關
    self._is_in_coast_window = False      # 是否處於滑行速度視窗內
    self._has_lead = False                # 是否有前車
    self._active_prev = False             # 前一週期是否啟動
    self._last_lead_time = 0.0            # 最後一次偵測到前車的時間

    self.active = False                   # 目前是否啟動滑行
    self.just_disabled = False            # 是否剛剛解除啟動

    self.current_ttc_threshold = 3.0      # 當前 TTC 門檻
    self.current_pitch = 0.0              # 當前車身縱傾角 (坡度)
    self.current_max_offset = 0.0         # 當前最大超速容許值

    self.personality = log.LongitudinalPersonality.standard 
    self._dtsc_is_active = False          # DTSC (彎道減速) 是否運作中

    self._soft_hold_factor = 1.0          # Soft Hold 平滑因數 (0.0~1.0)

  def _check_emergency_conditions(self, lead, v_ego, current_time):
    """檢查是否存在緊急狀況需要立即中止 ACM"""
    if not lead or not lead.status:
      return False

    closing_speed = max(v_ego - lead.vLead, 0.1)
    self.lead_ttc = lead.dRel / closing_speed 
    
    relative_speed = v_ego - lead.vLead         
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    # 觸發條件：TTC 過低、相對速差過大、或距離小於車速對應的最小安全距離
    if (self.lead_ttc < EMERGENCY_TTC) or \
       (relative_speed > EMERGENCY_RELATIVE_SPEED) or \
       (lead.dRel < min_dist_for_speed and relative_speed > 0):
      self._last_lead_time = current_time
      if self.active:
        cloudlog.warning(f"ACM emergency disable: dRel={lead.dRel:.1f}m, TTC={self.lead_ttc:.1f}s, RelSpeed={relative_speed:.1f}m/s")
      return True
    return False

  def _update_lead_status(self, lead, v_ego, current_time):
    """更新前車狀態與 TTC 資訊"""
    if lead and lead.status:
      closing_speed = max(v_ego - lead.vLead, 0.1)
      self.lead_ttc = lead.dRel / closing_speed
      
      self.current_ttc_threshold = np.interp(v_ego, TTC_BP, TTC_V) 
      if self.lead_ttc < self.current_ttc_threshold:
        self._has_lead = True
        self._last_lead_time = current_time
      else:
        self._has_lead = False
    else:
      self._has_lead = False
      self.lead_ttc = float('inf')

  def _check_cooldown(self, current_time):
    """檢查是否處於失去前車後的冷卻期"""
    time_since_lead = current_time - self._last_lead_time
    return time_since_lead < LEAD_COOLDOWN_TIME

  def _should_activate(self, user_ctrl_lon, v_ego, v_cruise, in_cooldown, pitch, dtsc_is_active):
    """判斷主動滑行 (ACM) 是否該啟動"""
    if dtsc_is_active: # 彎道減速中不滑行
        self._is_in_coast_window = False
        return False
    if pitch > PITCH_UPHILL_THRESHOLD: # 上坡不滑行，避免失速
        self._is_in_coast_window = False
        return False
    
    # 下坡視窗較嚴格，避免下滑太快
    if pitch < PITCH_DOWNHILL_THRESHOLD:
        self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH
    else:
        self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH

    lower_bound = v_cruise - (SPEED_OFFSET_MIN_KPH / 3.6)
    upper_bound = v_cruise + (self.current_max_offset / 3.6)
    self._is_in_coast_window = lower_bound < v_ego < upper_bound

    return (not user_ctrl_lon and     # 使用者未接管油門煞車
            not self._has_lead and    # 前方無緊跟之車輛
            not in_cooldown and       # 不在冷卻期
            self._is_in_coast_window) # 在速度容許範圍內

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, mode='acc', personality=log.LongitudinalPersonality.standard, dtsc_is_active=False):
    """主要狀態更新進入點"""
    self.personality = personality
    self._dtsc_is_active = dtsc_is_active 
    self._is_normal_mode = (mode == 'acc')

    if not self.enabled or len(cc.orientationNED) != 3:
      self.active = False
      return

    self.current_pitch = cc.orientationNED[1] 
    current_time = time.monotonic()
    lead = rs.leadOne

    # 先檢查緊急狀況，優先級最高
    if self._check_emergency_conditions(lead, v_ego, current_time):
      self.active = False
      self._active_prev = self.active
      return

    self._update_lead_status(lead, v_ego, current_time)
    in_cooldown = self._check_cooldown(current_time)
    self.active = self._should_activate(user_ctrl_lon, v_ego, v_cruise, in_cooldown, self.current_pitch, dtsc_is_active)

    # 記錄日誌
    self.just_disabled = self._active_prev and not self.active
    if self.active and not self._active_prev:
      pitch_deg = self.current_pitch * 57.2958
      cloudlog.info(f"ACM ON: v={v_ego*3.6:.0f}, pitch={pitch_deg:.1f}deg, Max+{self.current_max_offset:.0f}kph")
    elif self.just_disabled:
      cloudlog.info("ACM OFF")

    self._active_prev = self.active

  def _apply_soft_hold(self, a_desired_trajectory, v_ego, lead, t_follow):
    """
    Soft Hold 核心邏輯：
    1. 限制補油門力度，避免過度逼近前車。
    2. 當前車急煞或靜止時，提前切斷動力進行滑行緩衝。
    """
    # =========================================================================
    # 1. 狀態標記與判定
    # =========================================================================
    should_cancel_soft_hold = False
    
    if lead is None or not lead.status or lead.dRel > 100.0:
        should_cancel_soft_hold = True

    target_factor = 1.0   
    v_ego_kph = v_ego * 3.6
    current_soft_hold_accel = np.interp(v_ego_kph, SOFT_HOLD_SPEED_BP, SOFT_HOLD_ACCEL_V)

    is_lead_braking_strict = False

    if not should_cancel_soft_hold:
        # [優化點] 前車速度低於 3.6km/h (1.0m/s) 視同靜止、蠕行或即將煞停，強制觸發動力切斷
        is_lead_stopped = lead.vLead < 1.0  

        # 速域動態判定：涵蓋全速域，包含高速公路的靜止車防追尾機制
        if v_ego_kph <= 10.0:
            is_lead_braking_strict = lead.aLeadK < -0.1 or is_lead_stopped
        elif v_ego_kph <= 30.0:
            is_lead_braking_strict = lead.aLeadK < -0.5 or is_lead_stopped
        elif v_ego_kph <= 40.0:
            is_lead_braking_strict = lead.aLeadK < -1.0 or is_lead_stopped
        else: # 涵蓋 40 km/h 以上的所有高速域 (解鎖 50km/h 限制)
            is_lead_braking_strict = lead.aLeadK < -1.25 or is_lead_stopped

        closing_speed = max(v_ego - lead.vLead, 0.1)
        current_ttc = lead.dRel / closing_speed
        
        # 計算距離比例 (Ratio)：當前距離 / 舒適安全距離
        desired_dist = get_safe_obstacle_distance(v_ego, t_follow)
        lead_obstacle_dist = lead.dRel + get_stopped_equivalence_factor(lead.vLead)

        if desired_dist < 0.1:
            ratio = 10.0
        else:
            ratio = lead_obstacle_dist / desired_dist

        # 若前車駛離跟車範圍外，取消限制
        if ratio > 1.2:
            should_cancel_soft_hold = True

    # =========================================================================
    # 2. 核心控制因數計算
    # =========================================================================
    if should_cancel_soft_hold:
        target_factor = 1.0
        alpha = 0.40  # 快速釋放動力回歸正常
    else:
        distance_factor = 1.0 

        if self.current_pitch <= PITCH_UPHILL_THRESHOLD:
            # [優化點] 移除下限限制，避免與 MPC 減速牆發生控制間隙
            # 只要進入上限 0.99 以內且 TTC 危險，即視為需要開始緩衝
            if ratio < SOFT_HOLD_RANGE_MAX and current_ttc <= SOFT_HOLD_TTC_THRESHOLD:
                distance_factor = 0.0

        # 根據相對速度調整因數，前車加速離開時會快速恢復動力
        v_rel_factor = np.interp(lead.vRel, [-2.0, 0.5], [0.0, 1.0])
        target_factor = max(distance_factor, v_rel_factor)

        # 針對前車急煞或靜止的特殊壓制
        if ratio < SOFT_HOLD_RANGE_MAX:
            if is_lead_braking_strict: # 全速域適用，遇靜止車即強制壓制動力
                current_soft_hold_accel = 0.0
                target_factor = 0.0 

        # 根據目標因數是增加還是減少，決定平滑速度 (alpha)
        if target_factor > self._soft_hold_factor:
            alpha = 0.10 # 恢復動力時較緩慢，增加舒適度
        else:
            alpha = 0.20 # 限制動力時較快，保證安全性

    # 3. 執行 EMA (指數移動平均) 平滑過濾，避免控制量跳變
    self._soft_hold_factor = (1.0 - alpha) * self._soft_hold_factor + alpha * target_factor

    # 4. 輸出最終軌跡加速度
    if self._soft_hold_factor < 0.99:
        # 將當前 MPC 軌跡與 Soft Hold 的限制加速度進行加權混和
        dynamic_limit = np.maximum(a_desired_trajectory, 0.0) * self._soft_hold_factor + current_soft_hold_accel * (1.0 - self._soft_hold_factor)
        # 最終取最小值，確保不會干擾 MPC 的煞車需求
        a_desired_trajectory = np.minimum(a_desired_trajectory, dynamic_limit)

    return a_desired_trajectory

  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None, t_follow=None):
    """應用滑行與 Soft Hold 修改 MPC 加速度軌跡"""
    if getattr(self, '_dtsc_is_active', False) or \
       not getattr(self, '_is_normal_mode', True):
        return a_desired_trajectory

    traj = a_desired_trajectory

    # 處於 ACM 滑行啟動狀態
    if self.active:
      min_accel = np.min(traj)
      # 如果 MPC 請求強力煞車，中止滑行交還控制權
      if min_accel < EMERGENCY_DECEL_THRESHOLD:
        cloudlog.warning(f"ACM aborting: MPC requested {min_accel:.2f} m/s² braking")
        self.active = False
      else:
        # 在無前車時，將微弱的負加速度 (引擎煞車感) 抹平為 0.0 達成純滑行
        if not (lead is not None and lead.status):
          modified_trajectory = np.copy(traj)
          for i in range(len(modified_trajectory)):
            if -0.3 < modified_trajectory[i] < 0:
              modified_trajectory[i] = 0.0
          traj = modified_trajectory

    # 取得當前駕駛個性對應的跟車時間
    if t_follow is None:
        t_follow = get_T_FOLLOW(self.personality)

    # 最後套用 Soft Hold 柔和處理
    traj = self._apply_soft_hold(traj, v_ego, lead, t_follow)
    return traj
