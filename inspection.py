import cv2
import numpy as np

def inspect_gear(image_path):
    # 1. قراءة الصورة
    img = cv2.imread(image_path)
    if img is None:
        print("خطأ: لم يتم العثور على الصورة!")
        return

    # المرحلة الأولى: تنظيف الصورة (Phase 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # استخدمنا INV لعكس الألوان (الترس أبيض والخلفية سوداء)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # المرحلة الثانية: تحليل الشكل الهندسي (Topological Analysis)
    # 1. رسم الإطار الخارجي (Contours) وتجاهل التفاصيل الداخلية
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("لم يتم العثور على ترس.")
        return

    # اختيار الترس كأكبر شكل في الصورة
    main_contour = max(contours, key=cv2.contourArea)

    # 2. فكرة "الأستيك المطاطي" (Convex Hull) 
    # مطلوب استخدام returnPoints=False لحساب الفجوات لاحقاً
    hull = cv2.convexHull(main_contour, returnPoints=False)

    # 3. قياس الفجوات والكسور (Convexity Defects)
    defects = cv2.convexityDefects(main_contour, hull)

    status = "PASS" # الحالة الافتراضية إن الترس سليم
    
    if defects is not None:
        for i in range(defects.shape[0]):
            # استخراج بيانات الفجوة
            s, e, farthest_point_index, d_raw = defects[i, 0]
            
            # قسمة الناتج على 256 لحساب المسافة الحقيقية بالبيكسل
            actual_distance = d_raw / 256.0
            
            # المرحلة الثالثة: بوابة اتخاذ القرار (Tolerance Gate)
            # لو الفجوة عميقة (أكبر من 30 بيكسل مثلاً)، نعتبره كسر في السِنة
            if actual_distance > 150: 
                status = "FAIL: STRUCTURAL DEFECT DETECTED"
                
                # تحديد إحداثيات النقطة الأعمق في الكسر
                farthest_point = tuple(main_contour[farthest_point_index][0])
                
                # رسم مربع أحمر حول الكسر للفت انتباه المفتش
                cv2.rectangle(img, (farthest_point[0]-40, farthest_point[1]-40), 
                              (farthest_point[0]+40, farthest_point[1]+40), (0, 0, 255), 2)
                
                # إرسال إشارة رقمية للـ PLC لرفض القطعة (محاكاة)
                print(f"[PLC TRIGGER] Signal: {status} at Coordinates {farthest_point}")

    # كتابة النتيجة النهائية (PASS أو FAIL) على شاشة الكاميرا
    color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # عرض النتيجة النهائية للمفتش
    cv2.imshow("Final Automated Inspection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# تشغيل النظام
inspect_gear("gear_sample.jpg")