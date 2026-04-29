
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

# 速度門檻常數 (km/h 轉換為 m/s)
APM_DEPARTURE_SPEED = 30 * 1000 / 3600   # 30 km/h：起步激烈模式上限

# 場景 2 常數 (前車絕對速度、加速度與車距)
V_LEAD_RELAX_ENTER = 20 * 1000 / 3600    # 20 km/h：進入前車緩和模式的門檻，同時加入前車須減速狀態
A_LEAD_RELAX_ENTER = -0.1                # -0.1 m/s^2：前車處於減速狀態的門檻
V_LEAD_RELAX_EXIT = 22 * 1000 / 3600     # 22 km/h：解除前車緩和模式的門檻 
D_LEAD_RELAX_ENTER = 20.0                # 30 m：進入前車緩和模式的最短車距門檻 (場景2與場景3共用)

# 場景 3 常數 (與前車的相對速差)
V_REL_RELAX_ENTER = 20 * 1000 / 3600     # 20 km/h：自車比前車快 20 km/h 時，進入緩和模式
V_REL_RELAX_EXIT = 5 * 1000 / 3600       # 5 km/h：速差降至 5 km/h 以內 (速度差不多時)，解除緩和模式

V_EGO_STOPPED = 0.1                      # 低於 0.1 m/s (約 0.36 km/h) 視為完全靜止


class APM:

  def __init__(self):
    # 用來記憶車輛狀態
    self.is_departing = False       # 場景 1：是否在起步加速階段
    self.is_relaxed_mode = False    # 場景 2：是否正處於前車慢速的緩和模式
    self.is_approaching = False     # 場景 3：是否正快速接近慢車中 (速差過大)

  def get_personality(self, v_ego, has_lead, v_lead, a_lead, d_lead, personality):
    """
    參數說明:
    - v_ego: 自車速度 (m/s)
    - has_lead: 是否有前車 (布林值 True/False)
    - v_lead: 前車速度 (m/s)
    - a_lead: 前車加速度 (m/s^2)
    - d_lead: 與前車的距離 (m)
    - personality: 使用者原本設定的駕駛風格
    """
    
    # --- 1. 起步狀態更新 ---
    if v_ego < V_EGO_STOPPED:
      # 當車輛靜止時，標記為「準備起步」
      self.is_departing = True
    elif v_ego >= APM_DEPARTURE_SPEED:
      # 當車速超過 30 km/h，解除「起步階段」標記
      self.is_departing = False

    # --- 2. 判斷前車狀況 ---
    if has_lead:
      # 場景 2：前車絕對速度判斷 + 負加速判斷 + 車距判斷
      # 條件：前車 < 20 km/h 且 前車正在減速 (a_lead < A_LEAD_RELAX_ENTER) 且 車距大於等於 20 公尺
      if v_lead < V_LEAD_RELAX_ENTER and a_lead < A_LEAD_RELAX_ENTER and d_lead >= D_LEAD_RELAX_ENTER:
        self.is_relaxed_mode = True
      elif v_lead > V_LEAD_RELAX_EXIT:
        self.is_relaxed_mode = False
        
      # 場景 3：與前車相對速差判斷 + 車距判斷 (v_ego - v_lead 就是我比前車快多少)
      v_rel = v_ego - v_lead
      # 條件：我比前車快超過 20 km/h 且 車距大於等於 20 公尺 (新增車距條件)
      if v_rel >= V_REL_RELAX_ENTER and d_lead >= D_LEAD_RELAX_ENTER:
        self.is_approaching = True
      elif v_rel <= V_REL_RELAX_EXIT:
        # 速差縮小到 5 km/h 以內 (包含前車比我快的情形)，解除接近模式
        self.is_approaching = False
        
    else:
      # 如果沒有前車，解除所有與前車相關的緩和模式
      self.is_relaxed_mode = False
      self.is_approaching = False

    # --- 3. 決定最終輸出的模式 (依照優先級：場景 1 > 場景 2 > 場景 3) ---
    
    # 【最優先】場景 1：起步加速階段 (0~30 km/h)
    # 只要符合起步條件，無論前車狀態如何，一律輸出 aggressive
    if self.is_departing and v_ego < APM_DEPARTURE_SPEED:
      return log.LongitudinalPersonality.aggressive

    # 【次優先】場景 2：前車過慢或減速中
    # 在非起步狀態下，如果前車狀態符合場景 2，進入 relaxed 模式
    if self.is_relaxed_mode:
      return log.LongitudinalPersonality.relaxed

    # 【最後優先】場景 3：與前車的相對速差過大 (正在快速接近)
    # 如果場景 1 和場景 2 都不成立，才判斷是否符合場景 3
    if self.is_approaching:
      return log.LongitudinalPersonality.relaxed

    # 條件解除：切回原本的風格 (當上述場景都不成立時)
    return personality
