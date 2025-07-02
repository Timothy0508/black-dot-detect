import cv2
import numpy as np
import pandas as pd

def detect_black_dots(image_path, min_area=10, max_area=1000):
    """
    偵測照片中的黑點並記錄其座標。

    Args:
        image_path (str): 影像檔案的路徑。
        min_area (int): 最小的黑點面積（像素），小於此面積的會被忽略。
        max_area (int): 最大的黑點面積（像素），大於此面積的會被忽略。

    Returns:
        list: 包含每個黑點中心座標的列表，格式為 [(x1, y1), (x2, y2), ...]。
              如果沒有偵測到黑點，則返回空列表。
    """
    try:
        # 1. 載入影像
        image = cv2.imread(image_path)
        if image is None:
            print(f"錯誤：無法載入影像，請檢查路徑：{image_path}")
            return []
        resize_hight = int(image.shape[0]/4)
        resize_width = int(image.shape[1]/4)


        # 2. 灰階轉換
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. 二值化處理
        # 這裡使用OTSU二值化，它會自動計算最佳的閾值
        # 由於我們偵測黑點，所以是將亮度低的像素設為白色 (255)，亮度高的像素設為黑色 (0)
        # 這樣黑點在二值化後會變成白色，便於尋找輪廓
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. 形態學操作 (可選，用於去除雜訊或連接斷裂的點)
        # 開運算 (Opneing)：先侵蝕再膨脹，可以去除小的白色雜訊點
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. 輪廓偵測
        # cv2.findContours 會返回兩個值：輪廓列表和層次結構
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        black_dot_coordinates = []
        output_image = image.copy() # 複製一份影像用於繪圖顯示
        output_image = cv2.resize(output_image, (resize_width, int(resize_hight))) # 調整顯示大小

        # 遍歷每個偵測到的輪廓
        for contour in contours:
            # 計算輪廓的面積
            area = cv2.contourArea(contour)

            # 根據面積篩選黑點
            if min_area < area < max_area:
                # 計算輪廓的中心點（質心）
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    black_dot_coordinates.append((cX, cY))

                    # 在影像上繪製圓圈標示黑點，並顯示座標 (用於視覺化確認)
                    cv2.circle(output_image, (int(cX / 4), int(cY / 4)), 5, (0, 255, 0), -1) # 綠色圓點
                    cv2.putText(output_image, f"({cX},{cY})", (int((cX + 10) / 4), int((cY + 10) / 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1) # 紅色文字

        # 顯示處理後的影像 (可選)
        cv2.imshow("Original Image", image)
        cv2.imshow("Binary Image", binary_image)
        cv2.imshow("Detected Black Dots", output_image)
        cv2.waitKey(0) # 按任意鍵關閉視窗
        cv2.destroyAllWindows()

        return black_dot_coordinates

    except Exception as e:
        print(f"處理影像時發生錯誤：{e}")
        return []

image_file = 'test_images/test_image.jpg'
    # 範例：假設您的照片中黑點面積大約在 50 到 500 之間
dots = detect_black_dots(image_file, min_area=0, max_area=500)
print(f"偵測到 {len(dots)} 個黑點。")

if len(dots) != 0:
    print("偵測到的黑點座標：")
    for dot in dots:
        print(dot)

    df = pd.DataFrame(dots, columns=['X座標', 'Y座標'])
    output_file = "detected_black_dots.csv"
    df.to_csv(output_file, index=False)
else:
    print("未偵測到符合條件的黑點。")