import os
import time
from enum import Enum, auto

from openpilot.common.params import Params


# --- 系統與安全常數 ---
LOG_PATH = "/data/media/0/realdata/debug.log"
PARAM_REFRESH_SEC = 2.0
MIN_SPEED_MS = 0.1
# [安全鎖 4] 最高車速限制，約 35 km/h。超過此速度的 90 度大轉向屬危險動作，HTD 拒絕作動
MAX_SPEED_MS = 9.72


def _log(message: str) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f} {message}\n")
    except Exception:
        pass


class HTDState(Enum):
    INACTIVE = auto()
    MANUAL_TURN = auto()
    RAMPING = auto()


class HumanTurnDetection:
    def __init__(self) -> None:
        self._params = Params()
        self._last_params_read = 0.0

        # --- 1. UI 控制參數 (從 Params 讀取) ---
        self._enabled = False                # [修正1] 預設 False，等第一次 _read_params() 從 Params 讀取
        self._angle_threshold_deg = 60.0

        # --- 2. 系統內建參數 (不從 Params 讀取) ---
        self._angle_release_deg = 20.0
        self._torque_start_nm = 2.0
        self._torque_release_nm = 0.6
        self._resume_angle_lock_deg = 40.0   # 安全接管角度鎖：大於此值拒絕恢復自駕

        # --- 3. 狀態與計時器 ---
        self._state: HTDState = HTDState.INACTIVE
        self._state_change_time = 0.0
        self._trigger_start_time = 0.0

        # --- 4. 車輛動態資料快取 ---
        self._last_angle_raw = 0.0
        self._last_torque_raw = 0.0
        self._last_angle = 0.0
        self._last_torque = 0.0
        self._last_pressed = False

        # --- 5. 運行過程暫存變數 ---
        self._max_turn_angle = 0.0           # 本次彎道最大角度
        self._dynamic_delay = 0.5            # 準備接管時的動態等待秒數

    def _read_params(self) -> None:
        """定期更新外部設定參數"""
        now = time.monotonic()
        if now - self._last_params_read < PARAM_REFRESH_SEC:
            return
            
        self._last_params_read = now
        self._enabled = self._params.get_bool("dp_htd_enabled")
        self._angle_threshold_deg = self._get_float("dp_htd_turn_angle_threshold", 60.0)

    def _transition(self, new_state: HTDState, reason: str) -> None:
        """處理狀態切換並記錄 Log"""
        if new_state == self._state:
            return
            
        self._state = new_state
        self._state_change_time = time.monotonic()
        _log(
            f"HTD {new_state.name} reason={reason} angle={self._last_angle:.1f} "
            f"torque={self._last_torque:.2f} pressed={self._last_pressed} delay={self._dynamic_delay:.2f}"
        )

    def update(
        self,
        lat_active: bool,
        cruise_enabled: bool,
        steering_angle_deg: float,
        steering_torque_nm: float,
        v_ego: float,
        steering_pressed: bool = False,
    ) -> tuple[bool, HTDState]:
        
        self._read_params()

        # --- 更新車輛動態資料 ---
        self._last_angle_raw = steering_angle_deg
        self._last_torque_raw = steering_torque_nm
        self._last_angle = abs(steering_angle_deg)
        self._last_torque = abs(steering_torque_nm)
        self._last_pressed = steering_pressed

        # --- 守衛條件 (Guard Clauses) ---
        # [修正2] cruise_enabled 時 HTD 不應介入（巡航中不是「人工轉向」場景）
        # 同時保留速度範圍與啟用開關的守衛
        is_invalid_condition = (
            not self._enabled or 
            cruise_enabled or 
            not lat_active or 
            not (MIN_SPEED_MS <= v_ego <= MAX_SPEED_MS)
        )
        
        if is_invalid_condition:
            if self._state != HTDState.INACTIVE:
                self._transition(HTDState.INACTIVE, "disabled_or_invalid_condition")
            self._trigger_start_time = 0.0
            return True, self._state

        # --- 狀態機邏輯 (State Machine) ---
        if self._state == HTDState.INACTIVE:
            if self._should_trigger():
                self._max_turn_angle = self._last_angle
                self._transition(HTDState.MANUAL_TURN, "trigger")
                return False, self._state
            
        elif self._state == HTDState.MANUAL_TURN:
            self._max_turn_angle = max(self._max_turn_angle, self._last_angle)
            if self._should_release():
                # 計算動態延遲時間 (0.5 ~ 1.0 秒)
                calculated_delay = self._max_turn_angle / 270.0
                self._dynamic_delay = max(0.5, min(calculated_delay, 1.0))
                self._transition(HTDState.RAMPING, "release")
            return False, self._state

        elif self._state == HTDState.RAMPING:
            # [修正3] retrigger 判斷移至 RAMPING 最頂端，確保每個 tick 都能重新進入 MANUAL_TURN
            if self._should_trigger():
                self._trigger_start_time = 0.0  # [修正4] retrigger 時重置防抖計時器，避免立刻再觸發
                self._transition(HTDState.MANUAL_TURN, "retrigger")
                return False, self._state

            elapsed = time.monotonic() - self._state_change_time
            if elapsed >= self._dynamic_delay:
                # 角度尚未回正，繼續等待
                if self._last_angle > self._resume_angle_lock_deg:
                    return False, self._state

                # 滿足恢復條件
                self._max_turn_angle = 0.0
                self._transition(HTDState.INACTIVE, "resume")
                return True, self._state
                
            return False, self._state

        # 預設行為 (理論上不會走到這裡)
        return True, self._state

    def _should_trigger(self) -> bool:
        """判斷是否應該觸發手動轉向狀態"""
        direction_match = (self._last_angle_raw * self._last_torque_raw) > 0
        condition_met = (
            self._last_pressed
            and direction_match
            and self._last_torque >= self._torque_start_nm
            and self._last_angle >= self._angle_threshold_deg
        )

        if condition_met:
            if self._trigger_start_time == 0.0:
                self._trigger_start_time = time.monotonic()
            elif time.monotonic() - self._trigger_start_time >= 0.1:
                return True
        else:
            self._trigger_start_time = 0.0

        return False

    def _should_release(self) -> bool:
        """判斷是否應該解除手動轉向狀態"""
        # 軌道 A：方向盤回正且扭力歸零
        perfect_return = (
            self._last_torque <= self._torque_release_nm
            and self._last_angle <= self._angle_release_deg
        )
        # 軌道 B：駕駛放開方向盤
        hands_off = not self._last_pressed

        release_condition = perfect_return or hands_off

        if release_condition:
            self._trigger_start_time = 0.0  # 釋放時同步清除防抖計時

        return release_condition

    def _get_float(self, key: str, default: float) -> float:
        """安全地從 Params 取得浮點數"""
        try:
            val = self._params.get(key)
            return float(val) if val is not None else default
        except Exception:
            return default
