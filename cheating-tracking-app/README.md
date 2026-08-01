# 🎓 Cheating Tracking & Alert System

A desktop app (Tkinter) that tracks students in an exam video, detects
cheating behavior with a YOLO model + built-in ByteTrack tracking, draws a
red bounding box on suspected students (green for normal), saves a
screenshot + logs the event to a local SQLite database, and shows a
desktop notification banner.

## 🔗 Links

- **Training notebook (Kaggle):** [cheating-tracking](https://www.kaggle.com/code/anwernasr/cheating-tracking)

## Features

- **Upload Video** button — pick an exam video file
- Live bounding-box overlay: red = cheating, green = normal
- Screenshot auto-saved to `alerts/screenshots/` when cheating is detected
- Every event logged to `alerts/alerts.db` (SQLite): timestamp, behavior,
  confidence, screenshot path
- **Alert History** panel — double-click a row to open its screenshot
- On-screen notification banner when a new cheating event fires

## Running the app

```bash
pip install -r requirements.txt
python main.py
```

Place your trained YOLO weights as `best.pt` next to `main.py` before
running. Update `CHEATING_CLASS_NAME` in `main.py` if your model's class
name differs from `"cheating"`.

## Project structure

```
cheating-tracking-app/
├── main.py
├── requirements.txt
└── README.md
```

`alerts/` (screenshots + SQLite database) is created automatically on first run.

## License

MIT
