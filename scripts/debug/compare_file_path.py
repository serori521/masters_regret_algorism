import os
import csv
import math
from pathlib import Path

# 1. 基準となるディレクトリリスト
active_method_dirs = [
    'AMRD', 'AMRwc', 'AMRW', 'AMRWW', 'DMIN', 
    'E-AMRD', 'E-AMRW', 'E-AMRWW',  
    'E-MMRD', 'E-MMRW', 'E-MMRWW',  
    'E-DMIN', 'E-WMIN', 'E-WWMIN', 'EV', 
    'G-AMRD', 'G-AMRW', 'G-AMRWW', 
    'G-MMRD', 'G-MMRW', 'G-MMRWW', 
    'G-DMIN', 'G-WMIN', 'G-WWMIN', 'GM', 
    'MMRD',  'MMRwc', 'MMRW', 'MMRWW', 
    'DMIN', 'WMIN', 'WWMIN', 'WMIN', 
    'eAMRd', 'eAMRdc', 'eAMRw', 'eAMRwc', 
    'eMMRd', 'eMMRdc', 'eMMRw', 'eMMRwc', 
    'gAMRd', 'gAMRdc', 'gAMRw', 'gAMRwc', 
    'gMMRd', 'gMMRdc', 'gMMRw', 'gMMRwc'
]

# 2. 比較対象のファイル群（文字列として定義してリスト化）
files_raw = """
AMRD, E-AMRD, E-MMRW, G-AMRD, G-MMRW, WMIN, eMMRw, gMMRwc
AMRW, E-AMRW, E-MMRWW,G-AMRW, G-MMRWW,MMRD, WWMIN,eMMRwc,lAMRw
AMRWW,E-AMRWW,E-WMIN, G-AMRWW,G-WMIN, MMRW, gAMRw, lAMRwc
E-DMIN, E-WWMIN,G-DMIN, G-WWMIN,MMRWW,eAMRw,gAMRwc,lMMRw
DMIN, E-MMRD, EV, G-MMRD, GM, eAMRwc, gMMRw, lMMRwc
"""

# 改行をカンマに変換し、分割してリスト化（空白削除）
files = [x.strip() for x in files_raw.replace('\n', ',').split(',') if x.strip()]

# ---------------------------------------------------------
# 処理1: active_method_dirs にあるが、files にないもの (不足分: loss)
# ---------------------------------------------------------
loss = [item for item in active_method_dirs if item not in files]

# ---------------------------------------------------------
# 処理2: files にあるが、active_method_dirs にないもの (過剰分/未定義: extra)
# ---------------------------------------------------------
extra = [item for item in files if item not in active_method_dirs]

# 重複を除去して見やすくしたい場合は以下をコメントアウト解除してください
# extra = sorted(list(set(extra))) 

# 結果出力
print(f"--- 1. Missing in files (loss): {len(loss)} items ---")
print(loss)
print("\n" + "="*50 + "\n")
print(f"--- 2. Unknown in files (extra): {len(extra)} items ---")
print(extra)