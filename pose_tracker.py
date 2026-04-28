import cv2, mediapipe as mp, numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Mesh Avatar", cv2.WINDOW_NORMAL)
cv2.moveWindow("Mesh Avatar", 0, 0)  # pencereyi 1. ekranın sol üstüne taşı
cv2.setWindowProperty("Mesh Avatar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    canvas = np.zeros((h,w,3),dtype=np.uint8)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        for conn in mp_pose.POSE_CONNECTIONS:
            s,e = lm[conn[0]], lm[conn[1]]
            cv2.line(canvas,(int(s.x*w),int(s.y*h)),(int(e.x*w),int(e.y*h)),(0,255,0),2)

        for l in lm:
            cx,cy = int(l.x*w), int(l.y*h)
            cv2.circle(canvas,(cx,cy),3,(0,255,0),-1)

        fast_points = []
        for i,l in enumerate(lm):
            cx,cy = int(l.x*w), int(l.y*h)
            if i in prev_positions:
                vx = cx - prev_positions[i][0]
                vy = cy - prev_positions[i][1]
                speed = np.hypot(vx,vy)
                if speed > 35:
                    fast_points.append((cx,cy))
            prev_positions[i] = (cx,cy)

        for (px,py) in fast_points:
            cv2.circle(canvas,(px,py),6,(0,255,255),-1)

    cv2.imshow("Mesh Avatar",canvas)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()
