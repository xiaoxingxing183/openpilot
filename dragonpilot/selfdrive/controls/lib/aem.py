"""
Copyright (c) 2025, Rick Lan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, and/or sublicense,
for non-commercial purposes only, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.
- Commercial use (e.g. use in a product, service, or activity intended to
  generate revenue) is prohibited without explicit written permission from
  the copyright holder.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import time
import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants

# Cooldown times (how long to stay in experimental mode after trigger)
AEM_COOLDOWN_STOP = 0.5      # seconds - for stop sign/light detection
AEM_COOLDOWN_TTC = 2.0       # seconds - for lead TTC events
AEM_COOLDOWN_LANE = 0.5      # seconds - 💡 新增：車道線消失後的殘留維持時間

# Stop sign/light detection thresholds
SLOW_DOWN_BP = [0., 2.78, 5.56, 8.34, 11.12, 13.89, 15.28]
SLOW_DOWN_DIST = [10, 30., 50., 70., 80., 90., 120.]

# TTC-based triggering thresholds
TTC_THRESHOLD = 1.8          # seconds - trigger when TTC drops below this
MIN_SPEED_FOR_TTC = 5.0      # m/s (~18 km/h) - TTC meaningless at low speeds
MIN_CLOSING_SPEED = 0.5      # m/s - must be closing at least this fast

# Green light resume thresholds
STANDSTILL_THRESHOLD = 0.1   # m/s - considered stopped below this
GREEN_LIGHT_X_THRESHOLD = 30 # meters - model path must extend beyond this
GREEN_LIGHT_FRAMES = 5       # consecutive frames model must show clear path

# --- 自訂閾值設定 ---
SPEED_DISABLE_AEM = 70.0 / 3.6  # m/s (70 km/h) - 超過此速度關閉 AEM
SPEED_ENABLE_AEM = 65.0 / 3.6   # m/s (65 km/h) - 低於此速度重新開啟 AEM
LEAD_DIST_FORCE_ACC = 8.0       # meters - 前車距離低於此值時，強制保持 ACC 模式
LANE_CONF_THRESHOLD = 0.3       # 車道線信心度低於 30% 觸發實驗模式


class AEM:

  def __init__(self):
    self._active = False
    self._cooldown_end_time = 0.0
    self._green_light_counter = 0
    
    # 狀態變數
    self._speed_allowed = True   
    self._force_acc = False      

  def _perform_experimental_mode(self, cooldown: float = AEM_COOLDOWN_TTC):
    self._active = True
    # Extend cooldown if new trigger comes in
    new_end = time.monotonic() + cooldown
    self._cooldown_end_time = max(self._cooldown_end_time, new_end)

  def get_mode(self, mode):
    # 檢查當下是否正在處理紅綠燈、急煞，或是「車道線剛恢復的 1.5 秒內」
    is_active_event = time.monotonic() < self._cooldown_end_time

    # 1. 高速限制 (最高優先權)：大於 70 km/h 直接關閉 AEM
    if not self._speed_allowed:
      self._active = False
      return mode

    # 2. 距離限制：如果在 8m 內，但「沒有」正在處理任何事件，才強制 ACC
    # (如果正在處理車道線模糊，此防線會被安全地繞過，維持實驗模式)
    if self._force_acc and not is_active_event:
      self._active = False
      return mode

    # 3. 正常觸發邏輯：若事件仍在維持時間內，進入 blended
    if is_active_event:
      mode = 'blended'
    else:
      self._active = False
      
    return mode

  def update_states(self, model_msg, radar_msg, v_ego):
    # 1. 更新延遲開關狀態
    if v_ego > SPEED_DISABLE_AEM:
      self._speed_allowed = False
    elif v_ego < SPEED_ENABLE_AEM:
      self._speed_allowed = True

    # 2. 更新強制 ACC 狀態
    self._force_acc = radar_msg.leadOne.status and radar_msg.leadOne.dRel <= LEAD_DIST_FORCE_ACC

    # 3. 更新車道線信心度 (加入 1.5 秒視覺殘留延遲)
    if len(model_msg.laneLineProbs) >= 3:
      l_prob = model_msg.laneLineProbs[1]
      r_prob = model_msg.laneLineProbs[2]
      if (l_prob < LANE_CONF_THRESHOLD) or (r_prob < LANE_CONF_THRESHOLD):
        # 只要有一瞬間沒線，就強制維持 1.5 秒的實驗模式
        self._perform_experimental_mode(AEM_COOLDOWN_LANE)
    else:
      self._perform_experimental_mode(AEM_COOLDOWN_LANE)

    # =========================================================
    # Stop sign/light detection (台灣市區闖燈刺客防護版)
    # =========================================================
    if len(model_msg.orientation.x) == len(model_msg.position.x) == ModelConstants.IDX_N:
      
      expected_stop_dist = np.interp(v_ego, SLOW_DOWN_BP, SLOW_DOWN_DIST)
      model_end_x = model_msg.position.x[ModelConstants.IDX_N - 1]
      
      # 模型預測需要煞車 (路徑被截斷)
      if model_end_x < expected_stop_dist:
        
        has_lead = radar_msg.leadOne.status
        lead_dist = radar_msg.leadOne.dRel if has_lead else 999.0
        v_rel = radar_msg.leadOne.vRel if has_lead else 0.0
        v_lead = radar_msg.leadOne.vLead if has_lead else 0.0
        
        # 🛡️ 基礎動態緩衝：前車已經完全離開 (距離拉得很遠)
        dynamic_buffer = max(8.0, v_ego * 0.3)
        
        # 🚨 闖燈刺客偵測 (Runner Detection)
        is_runner = has_lead and (v_lead > 5.5) and (v_rel > -0.5) and (lead_dist > model_end_x + 4.0)
        
        # 觸發條件：沒車 / 前車跑遠了 / 判定前車正在闖紅燈！
        if not has_lead or (lead_dist > model_end_x + dynamic_buffer) or is_runner:
            self._perform_experimental_mode(AEM_COOLDOWN_STOP)
    # =========================================================

    # Green light resume
    if self._active and v_ego < STANDSTILL_THRESHOLD and not radar_msg.leadOne.status and \
      len(model_msg.position.x) == ModelConstants.IDX_N and \
      model_msg.position.x[ModelConstants.IDX_N - 1] > GREEN_LIGHT_X_THRESHOLD:
      self._green_light_counter += 1
      if self._green_light_counter >= GREEN_LIGHT_FRAMES:
        self._cooldown_end_time = 0.0
    else:
      self._green_light_counter = 0

    # TTC-based triggering
    if v_ego > MIN_SPEED_FOR_TTC and radar_msg.leadOne.status:
      closing_speed = -radar_msg.leadOne.vRel
      if closing_speed > MIN_CLOSING_SPEED:
        d_rel = radar_msg.leadOne.dRel
        if d_rel > 0:
          ttc = d_rel / closing_speed
          if ttc < TTC_THRESHOLD:
            self._perform_experimental_mode(AEM_COOLDOWN_TTC)
