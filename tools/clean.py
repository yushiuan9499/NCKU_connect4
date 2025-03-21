# 這個python程式碼是用來清除logs當中得學號
import sys
import glob
import os
import re

# 取得檔名（不含路徑）
def get_filename(path):
    return path.split("/")[-1]

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 rm.py [zipname]")
        return
    # 取得logs 位於的資料夾
    path = sys.argv[1]
    # 創建新的資料夾
    os.makedirs(f"{path}_cleaned", exist_ok=False)
    # 統計現在到第幾個學號
    cnt = 0
    # 學號對編號的mapping
    id_to_num = {}
    for filename in glob.glob(f"{path}/*.txt"):
        with open(filename, "r") as old_f:
            filename = get_filename(filename)
            # 檔名會是A12345678_vs_B12345678.txt
            # 所以這些學號也要移掉
            if filename[0:9] not in id_to_num:
                id_to_num[filename[0:9]] = cnt
                cnt += 1
            if filename[13:22] not in id_to_num:
                id_to_num[filename[13:22]] = cnt
                cnt += 1
            new_filename = "Student" + str(id_to_num[filename[0:9]]) + "_vs_" + "Student" + str(id_to_num[filename[13:22]]) + ".txt"
            with open(f"{path}_cleaned/{new_filename}", "w") as new_f:
                while True:
                    line = old_f.readline()
                    if not line:
                        break
                    # 找到所有學號，規則是一個大寫字母加上1個大寫字母或是數字最後加上7個數字
                    ids = re.findall(r"[A-Z][A-Z0-9]\d{7}", line)
                    # 如果有找到學號，則將學號替換成"Student"+編號
                    for id in ids:
                        if id not in id_to_num:
                            id_to_num[id] = cnt
                            cnt += 1
                        line = line.replace(id, "Student" + str(id_to_num[id]))
                    # 將處理過的line寫入新的檔案
                    new_f.write(line)

if __name__ == "__main__":
    main()
