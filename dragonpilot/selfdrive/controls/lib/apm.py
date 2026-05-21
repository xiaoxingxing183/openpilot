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
    # 用來記憶車輛狀態
    self.is_departing = False       # 場景 1：是否在起步加速階段
    self.is_relaxed_mode = False    # 場景 2：是否正處於前車慢速的緩和模式
    self.is_approaching = False     # 場景 3：是否正快速接近慢車中 (速差過大)
    
    # 濾波平滑化狀態
    self.v_rel_smoothed = None      # 用來儲存過濾/平滑化後的相對速差

  def get_personality(self, v_ego, has_lead, v_lead, a_lead, d_lead, personality, t_follow_relaxed=1.75):
    """
    參數說明:
    - v_ego: 自車速度 (m/s)
    - has_lead: 是否有前車 (布林值 True/False)
    - v_lead: 前車速度 (m/s)
    - a_lead: 前車加速度 (m/s^2)
    - d_lead: 與前車的距離 (m)
    - personality: 使用者原本設定的駕駛風格
    - t_follow_relaxed: relaxed 模式下的跟車時間 (秒)，用來換算動態距離門檻
    """
    
    # --- 1. 起步狀態更新 ---
    if v_ego < V_EGO_STOPPED:
      # 當車輛靜止時，標記為「準備起步」
      self.is_departing = True
    elif v_ego >= APM_DEPARTURE_SPEED:
      # 當車速超過 15 km/h，解除「起步階段」標記
      self.is_departing = False

    # --- 2. 判斷前車狀況 ---
    if has_lead:
      # 取得當下瞬間的相對速差 (原始含有雜訊的資料)
      v_rel_raw = v_ego - v_lead

      # --- 低通濾波處理 (Exponential Moving Average) ---
      if self.v_rel_smoothed is None:
        # 剛抓到前車時，以當下數值為基準
        self.v_rel_smoothed = v_rel_raw
      else:
        # 保留 90% 歷史平滑值，採納 10% 最新值，削平雜訊與突波
        self.v_rel_smoothed = self.v_rel_smoothed * 0.90 + v_rel_raw * 0.10
      
      # 將平滑後的數值指派給 v_rel，供後續判定使用
      v_rel = self.v_rel_smoothed

      # 計算動態距離門檻 (秒數換算距離，並設定 10m 為最低下限，避免塞車時過於敏感)
      d_req = max(10.0, v_ego * t_follow_relaxed)

      # 場景 2：前車絕對速度判斷 + 負加速判斷 + 車距大於動態門檻
      if v_lead < V_LEAD_RELAX_ENTER and a_lead < A_LEAD_RELAX_ENTER and d_lead >= d_req:
        self.is_relaxed_mode = True
      # 當速差降至 10 km/h 以內時解除
      elif v_rel <= V_REL_RELAX_EXIT:
        self.is_relaxed_mode = False
        
      # 場景 3：與前車相對速差判斷 + 車距大於動態門檻
      if v_rel >= V_REL_RELAX_ENTER and d_lead >= d_req:
        self.is_approaching = True
      # 當速差降至 10 km/h 以內時解除
      elif v_rel <= V_REL_RELAX_EXIT:
        self.is_approaching = False
        
    else:
      # 如果沒有前車，解除所有與前車相關的緩和模式
      self.is_relaxed_mode = False
      self.is_approaching = False
      self.v_rel_smoothed = None  # 失去前車目標時，清空平滑化數值

    # --- 3. 決定最終輸出的模式 (依照優先級：場景 1 > 場景 2 > 場景 3) ---
    
    # 【最優先】場景 1：起步加速階段 (0~15 km/h)
    if self.is_departing and v_ego < APM_DEPARTURE_SPEED:
      return log.LongitudinalPersonality.aggressive

    # 【次優先】場景 2：前車過慢或減速中
    if self.is_relaxed_mode:
      return log.LongitudinalPersonality.relaxed

    # 【最後優先】場景 3：與前車的相對速差過大 (正在快速接近)
    if self.is_approaching:
      return log.LongitudinalPersonality.relaxed

    # 條件解除：切回原本的風格
    return personality
