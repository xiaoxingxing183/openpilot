"""
Copyright (c) 2026, Rick Lan

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

from cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

# 速度門檻常數 (km/h 轉換為 m/s)
APM_DEPARTURE_SPEED = 15 * 1000 / 3600   # 15 km/h：起步激烈模式上限

# 場景 2 常數 (前車絕對速度、加速度)
V_LEAD_RELAX_ENTER = 20 * 1000 / 3600    # 20 km/h：進入前車緩和模式的門檻，同時加入前車須減速狀態
A_LEAD_RELAX_ENTER = -0.2                # -0.2 m/s^2：前車處於減速狀態的門檻

# 場景 3 常數 (與前車的相對速差)
V_REL_RELAX_ENTER = 20 * 1000 / 3600     # 20 km/h：自車比前車快 20 km/h 時，進入緩和模式
V_REL_RELAX_EXIT = 10 * 1000 / 3600      # 10 km/h：速差降至 10 km/h 以內 (速度差不多時)，解除緩和模式 (場景2與3共用)

V_EGO_STOPPED = 0.5                      # 低於 0.5 m/s (約 1.8 km/h) 視為完全靜止


class APM:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    
    # 用來記憶車輛狀態
    self.is_departing = False       # 場景 1：是否在起步加速階段
    self.is_relaxed_mode = False    # 場景 2：是否正處於前車慢速的緩和模式
    self.is_approaching = False     # 場景 3：是否正快速接近慢車中 (速差過大)
    
    # 濾波平滑化狀態
    self.v_rel_smoothed = None      # 用來儲存過濾/平滑化後的相對速差

    # 初始化開關狀態
    self._enabled = self.params.get_bool("dp_lon_apm")

  def update(self, sm=None):
    """依照系統物理週期更新 Params，參照 accel_controller"""
    self.frame += 1
    # 利用 DT_MDL 換算真實物理時間，精準每 10.0 秒讀取一次 Params
    if self.frame % int(10.0 / DT_MDL) == 0:
      self._enabled = self.params.get_bool("dp_lon_apm")

  def is_enabled(self) -> bool:
    return self._enabled

  def get_personality(self, v_ego, has_lead, v_lead, a_lead, d_lead, personality, t_follow_relaxed=1.75):
    # 如果模組未啟用，直接攔截並回傳原廠設定風格，不消耗任何計算資源
    if not self._enabled:
      return personality
      
    # --- 1. 起步狀態更新 ---
    if v_ego < V_EGO_STOPPED:
      self.is_departing = True
    elif v_ego >= APM_DEPARTURE_SPEED:
      self.is_departing = False

    # --- 2. 判斷前車狀況 ---
    if has_lead:
      v_rel_raw = v_ego - v_lead

      # --- 低通濾波處理 (Exponential Moving Average) ---
      if self.v_rel_smoothed is None:
        self.v_rel_smoothed = v_rel_raw
      else:
        self.v_rel_smoothed = self.v_rel_smoothed * 0.90 + v_rel_raw * 0.10
      
      v_rel = self.v_rel_smoothed
      
      # 已經將最低距離門檻從 10.0 修改為 30.0
      d_req = max(30.0, v_ego * t_follow_relaxed)

      # 場景 2：前車絕對速度判斷 + 負加速判斷 + 車距大於動態門檻
      if v_lead < V_LEAD_RELAX_ENTER and a_lead < A_LEAD_RELAX_ENTER and d_lead >= d_req:
        self.is_relaxed_mode = True
      elif v_rel <= V_REL_RELAX_EXIT:
        self.is_relaxed_mode = False
        
      # 場景 3：與前車相對速差判斷 + 車距大於動態門檻
      if v_rel >= V_REL_RELAX_ENTER and d_lead >= d_req:
        self.is_approaching = True
      elif v_rel <= V_REL_RELAX_EXIT:
        self.is_approaching = False
        
    else:
      self.is_relaxed_mode = False
      self.is_approaching = False
      self.v_rel_smoothed = None  

    # --- 3. 決定最終輸出的模式 (優先級：場景 1 > 場景 2 > 場景 3) ---
    if self.is_departing and v_ego < APM_DEPARTURE_SPEED:
      return log.LongitudinalPersonality.aggressive

    if self.is_relaxed_mode:
      return log.LongitudinalPersonality.relaxed

    if self.is_approaching:
      return log.LongitudinalPersonality.relaxed

    return personality
