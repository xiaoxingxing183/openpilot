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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

class DASR:

  def __init__(self):
    # 分別定義正加速(油門)與負加速(煞車)的動態斜率預設值
    self.slew_rate_up = 0.05    # 預設正向加速變化率
    self.slew_rate_down = 0.05  # 預設負向減速變化率
    
    # 記錄車輛是否處於「從靜止起步」的狀態 (自帶防跳動：1.8開 / 15關)
    self.is_taking_off = False
    
    # --- 新增各場景的防跳動狀態標記 ---
    self.high_speed_active = False      # 高速防跳動 (80開 / 70關)
    self.mid_speed_active = False       # 中速防跳動 (30開 / 20關)
    self.low_speed_brake_active = False # 低速煞停防跳動 (5~10開 / <3或>12關)

  def update(self, v_ego, a_target, has_lead=False):
    """
    更新動態斜率狀態
    :param v_ego: 目前車速 (m/s)
    :param a_target: 目標加速度 (m/s^2)，大於 0 為正加速，小於等於 0 為負加速
    :param has_lead: 前方是否有車 (Boolean)
    """
    # ==========================================
    # 狀態更新區 (計算防跳動區間)
    # ==========================================
    # 場景 1 防跳動：起步 (低於 1.8 km/h 開啟，高於 15 km/h 關閉)
    if v_ego < 0.5:
      self.is_taking_off = True
    elif v_ego > 4.17:
      self.is_taking_off = False

    # 場景 2 防跳動：高速 (高於 80 km/h 開啟，低於 70 km/h 關閉)
    if v_ego >= 22.22:
      self.high_speed_active = True
    elif v_ego < 19.44:
      self.high_speed_active = False

    # 場景 4 防跳動：中速 (高於 30 km/h 開啟，低於 20 km/h 關閉)
    if v_ego >= 8.33:
      self.mid_speed_active = True
    elif v_ego < 5.56:
      self.mid_speed_active = False

    # 場景 3 防跳動：低速煞停 (進入 5~10 km/h 開啟，低於 3 km/h 或 高於 12 km/h 關閉)
    if 1.39 <= v_ego <= 2.78:
      self.low_speed_brake_active = True
    elif v_ego < 0.83 or v_ego > 3.33:
      self.low_speed_brake_active = False

    # ==========================================
    # 基礎預設值 (所有車系、所有狀態的預設底線)
    # ==========================================
    self.slew_rate_up = 0.05
    self.slew_rate_down = 0.05

    # ==========================================
    # 場景 4：中高速跟車場景
    # 條件：【中速區間 (30開/20關)】 且 【前方有車 (has_lead)】 且 【正加速】
    # ==========================================
    if self.mid_speed_active and has_lead and a_target > 0.0:
      self.slew_rate_up = 0.04   # 中高速且有前車時，正加速限制得更平緩

    # ==========================================
    # 場景 2：高速巡航場景
    # 條件：【高速區間 (80開/70關)】 且 【正加速】
    # ==========================================
    if self.high_speed_active and a_target > 0.0:
      self.slew_rate_up = 0.03  # 高速正加速更平緩，避免突兀推背感 (會蓋過場景4)

    # ==========================================
    # 場景 3：低速快煞停場景
    # 條件：【低速煞停區間 (5~10開)】 且 【負加速】
    # ==========================================
    if self.low_speed_brake_active and a_target <= 0.0:
      self.slew_rate_down = 0.04 # 快煞停時煞車變化更柔和，減少點頭頓挫感

    # ==========================================
    # 場景 1：靜止起步場景 (最高優先權！)
    # 條件：【所有車系】 且 【處於起步狀態 (1.8開/15關)】 且 【正加速】
    # ==========================================
    if self.is_taking_off and a_target > 0.0:
      self.slew_rate_up = 0.06   # 提升起步靈敏度
