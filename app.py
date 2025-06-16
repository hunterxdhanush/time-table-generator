import streamlit as st
from datetime import time, timedelta, datetime
import pandas as pd
from timetable_csp import TimetableCSP
import io
from collections import Counter
import math

# --- Custom CSS to spread out content but not edge-to-edge, and retain black background ---
st.markdown(
    """
    <style>
    html, body, .main, .block-container {
        max-width: 1400px !important;
        width: 100% !important;
        min-width: 0 !important;
        margin-left: auto !important;
        margin-right: auto !important;
        /* background: #fff;  Remove forced white background */
    }
    .main .block-container {
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1400px !important;
        width: 100% !important;
    }
    .stDataFrame, .stTable, .stMarkdown table, .stMarkdown th, .stMarkdown td {
        width: 100% !important;
        max-width: 100vw;
        min-width: 0 !important;
        text-align: center;
    }
    .stMarkdown table {
        table-layout: fixed;
        width: 100% !important;
    }
    .stMarkdown th, .stMarkdown td {
        word-break: break-word;
    }
    .css-1kyxreq, .css-1v0mbdj, .stTextInput, .stNumberInput, .stSelectbox, .stMultiSelect {
        width: 100% !important;
        max-width: 100vw;
        min-width: 0 !important;
    }
    .stExpanderContent {
        width: 100% !important;
        max-width: 100vw !important;
        min-width: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Timetable Generator")

# --- Custom time slots ---
CUSTOM_SLOTS = [
    (time(8, 0), time(9, 0)),
    (time(9, 0), time(10, 0)),
    (time(10, 30), time(11, 30)),
    (time(11, 30), time(12, 30)),
    (time(14, 0), time(15, 0)),
    (time(15, 0), time(16, 0)),
    (time(16, 0), time(17, 0)),
]
SLOT_LABELS = [
    "8:00-9:00",
    "9:00-10:00",
    "10:30-11:30",
    "11:30-12:30",
    "14:00-15:00",
    "15:00-16:00",
    "16:00-17:00",
]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# --- Helper for 12hr time format ---
def to_12hr(t):
    return t.strftime('%I:%M %p').lstrip('0').replace(' 0', ' ')

# Custom slot labels in 12hr format
SLOT_LABELS_12HR = [f"{to_12hr(slot[0])} - {to_12hr(slot[1])}" for slot in CUSTOM_SLOTS]
# Insert tea break and lunch break as columns at correct positions
BREAK_COLUMNS = [
    ("10:00 AM - 10:30 AM", "Tea Break", 2),  # after 9:00-10:00
    ("12:30 PM - 2:00 PM", "Lunch Break", 4), # after 11:30-12:30
]
all_columns = SLOT_LABELS_12HR.copy()
for break_time, break_name, idx in sorted(BREAK_COLUMNS, key=lambda x: -x[2]):
    all_columns.insert(idx, f"{break_name}\n({break_time})")

# --- Session State for config persistence ---
if 'last_teachers' not in st.session_state:
    st.session_state['last_teachers'] = None
if 'last_classes' not in st.session_state:
    st.session_state['last_classes'] = None
if 'timetable_generated' not in st.session_state:
    st.session_state.timetable_generated = False

# --- Load last config ---
if st.button("Load Last Configuration"):
    if st.session_state['last_teachers'] and st.session_state['last_classes']:
        teachers = st.session_state['last_teachers']
        classes = st.session_state['last_classes']
        st.success("Last configuration loaded!")
    else:
        teachers = []
        classes = []
        st.info("No configuration saved yet.")
else:
    teachers = []
    classes = []

# --- Input: Number of classes (needed for teacher class selection) ---
num_classes = st.number_input("Number of classes", min_value=1, step=1, value=len(classes) or 1)
class_names_for_teacher = [classes[i]["name"] if i < len(classes) else f"Class {i+1}" for i in range(int(num_classes))]

# --- Input: Number of teachers ---
num_teachers = st.number_input("Number of teachers", min_value=1, step=1, value=len(teachers) or 1)
st.subheader("Teacher Details")
for i in range(int(num_teachers)):
    with st.expander(f"Teacher {i+1}"):
        name = st.text_input(f"Name", key=f"teacher_name_{i}", value=teachers[i]["name"] if i < len(teachers) else "")
        num_subjects = st.number_input(f"Number of subjects for {name or f'Teacher {i+1}'}", min_value=1, step=1, key=f"teacher_num_subjects_{i}", value=len(teachers[i]["subjects"]) if i < len(teachers) else 1)
        subjects = []
        for s in range(int(num_subjects)):
            subject = st.text_input(f"Subject {s+1}", key=f"teacher_{i}_subject_{s}", value=teachers[i]["subjects"][s] if i < len(teachers) and s < len(teachers[i]["subjects"]) else "")
            if subject:
                subjects.append(subject)
        col1, col2 = st.columns(2)
        with col1:
            start_hour = st.number_input(f"Start hour (24h)", min_value=0, max_value=23, value=teachers[i]["working_hours"][0] if i < len(teachers) else 9, key=f"teacher_start_{i}")
        with col2:
            end_hour = st.number_input(f"End hour (24h)", min_value=0, max_value=23, value=teachers[i]["working_hours"][1] if i < len(teachers) else 15, key=f"teacher_end_{i}")
        working_hours = (start_hour, end_hour)
        if i < len(teachers):
            teachers[i] = {"name": name, "subjects": subjects, "working_hours": working_hours}
        else:
            teachers.append({"name": name, "subjects": subjects, "working_hours": working_hours})

# --- Input: Class Details ---
st.subheader("Class Details")
for i in range(int(num_classes)):
    with st.expander(f"Class {i+1}"):
        class_name = st.text_input(f"Class Name", key=f"class_name_{i}", value=classes[i]["name"] if i < len(classes) else "")
        num_subjects = st.number_input(f"Number of subjects for {class_name or f'Class {i+1}'}", min_value=1, step=1, key=f"class_num_subjects_{i}", value=len(classes[i]["subjects"]) if i < len(classes) and isinstance(classes[i]["subjects"], list) else 1)
        class_subjects = []
        for s in range(int(num_subjects)):
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                subject = st.text_input(f"Subject {s+1} Name", key=f"class_{i}_subject_{s}_name", value=classes[i]["subjects"][s]["name"] if i < len(classes) and s < len(classes[i]["subjects"]) and isinstance(classes[i]["subjects"][s], dict) else "")
            with col2:
                teacher_options = [t["name"] for t in teachers if t["name"]]
                teacher = st.selectbox(f"Teacher for {subject or f'Subject {s+1}'}", teacher_options, key=f"class_{i}_subject_{s}_teacher", index=teacher_options.index(classes[i]["subjects"][s]["teacher"]) if i < len(classes) and s < len(classes[i]["subjects"]) and isinstance(classes[i]["subjects"][s], dict) and classes[i]["subjects"][s]["teacher"] in teacher_options else 0)
            with col3:
                periods = st.number_input(f"Periods", min_value=1, step=1, key=f"class_{i}_subject_{s}_periods", value=classes[i]["subjects"][s]["periods"] if i < len(classes) and s < len(classes[i]["subjects"]) and isinstance(classes[i]["subjects"][s], dict) and "periods" in classes[i]["subjects"][s] else 1)
            if subject and teacher:
                class_subjects.append({"name": subject, "teacher": teacher, "periods": periods})
        if i < len(classes):
            classes[i] = {"name": class_name, "subjects": class_subjects}
        else:
            classes.append({"name": class_name, "subjects": class_subjects})

# --- Save config for reload ---
if st.button("Save Current Configuration"):
    st.session_state['last_teachers'] = teachers.copy()
    st.session_state['last_classes'] = classes.copy()
    st.success("Configuration saved!")

# --- Display collected data for confirmation as tables ---
st.subheader("Collected Teachers Data (Table)")
if teachers:
    teachers_df = pd.DataFrame([
        {
            "Name": t["name"],
            "Subjects": ", ".join(t["subjects"]),
            "Working Hours": f"{t['working_hours'][0]}:00 - {t['working_hours'][1]}:00"
        }
        for t in teachers
    ])
    st.dataframe(teachers_df)
else:
    st.info("No teachers data to display.")

st.subheader("Collected Classes Data (Table)")
if classes:
    classes_df = pd.DataFrame([
        {
            "Class Name": c["name"],
            "Subjects": ", ".join(f"{s['name']} ({s['teacher']}, {s['periods']} periods)" for s in c["subjects"])
        }
        for c in classes
    ])
    st.dataframe(classes_df)
else:
    st.info("No classes data to display.")

# --- Helper functions to ensure uniqueness ---
def make_unique(names):
    counts = Counter()
    result = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            result.append(name)
        else:
            result.append(f"{name} ({counts[name]})")
    return result

def make_unique_labels(labels):
    counts = Counter()
    result = []
    for label in labels:
        counts[label] += 1
        if counts[label] == 1:
            result.append(label)
        else:
            result.append(f"{label} ({counts[label]})")
    return result

# --- Validation: total periods per class vs available slots ---
def get_total_slots_per_week():
    return len(CUSTOM_SLOTS) * len(DAYS)

# --- Generate Timetable Button ---
if st.button("Generate Timetable"):
    # Validation: total periods per class
    total_slots_per_week = get_total_slots_per_week()
    for c in classes:
        total_periods = sum(int(subj["periods"]) for subj in c["subjects"])
        if total_periods > total_slots_per_week:
            st.error(f"Class '{c['name']}' requests {total_periods} periods, but only {total_slots_per_week} slots are available in a week.")
            st.stop()
    # Prepare data for CSP solver
    class_names = [c["name"] for c in classes]
    # Build teacher dicts for CSP
    teacher_objs = []
    for t in teachers:
        start = time(int(t["working_hours"][0]), 0)
        end = time(int(t["working_hours"][1]), 0)
        teacher_objs.append({"name": t["name"], "start_time": start, "end_time": end, "subjects": t["subjects"]})
    # Build subjects structure for CSP: for each class, list of dicts {subject, teacher, periods}
    subjects = []
    for c in classes:
        class_subjects = []
        for subj in c["subjects"]:
            for _ in range(int(subj["periods"])):
                class_subjects.append({"name": subj["name"], "teacher": subj["teacher"], "periods": subj["periods"]})
        subjects.append(class_subjects)

    # Run CSP for the whole week
    csp = TimetableCSP(class_names, teacher_objs, subjects, slot_minutes=60, custom_slots=CUSTOM_SLOTS, days=DAYS)
    
    timetable = csp.solve()
    if timetable is None:
        st.error("No valid timetable could be generated with the given constraints.")
        st.session_state.timetable_generated = False
    else:
        st.session_state.timetable_generated = True
        # Collect all unique (day, slot) pairs
        all_slots = set()
        for class_entries in timetable.values():
            for entry in class_entries:
                all_slots.add(entry["slot"])
        all_slots = sorted(list(all_slots), key=lambda x: (DAYS.index(x[0]), x[1][0], x[1][1]))
        # Ensure unique class names and slot labels
        unique_class_names = make_unique(class_names)
        slot_labels = [f"{day} {slot[0].strftime('%H:%M')} - {slot[1].strftime('%H:%M')}" for (day, slot) in all_slots]
        unique_slot_labels = make_unique_labels(slot_labels)
        # Build DataFrame: rows=unique_class_names, columns=unique_slot_labels
        data = {label: [] for label in unique_slot_labels}
        for (day_slot), col_label in zip(all_slots, unique_slot_labels):
            for idx, class_name in enumerate(unique_class_names):
                orig_class_name = class_names[idx]
                entry = next((e for e in timetable[orig_class_name] if e["slot"] == day_slot), None)
                if entry:
                    data[col_label].append(f"{entry['subject']} - {entry['teacher']}")
                else:
                    data[col_label].append("")
        df = pd.DataFrame(data, index=unique_class_names)
        st.subheader("Generated Timetable (All Slots, Whole Week)")
        # --- Color coding subjects ---
        def subject_color(val):
            if not val or '-' not in val:
                return ''
            subject = val.split('-')[0].strip()
            colors = ['#FFDDC1', '#C1FFD7', '#C1D4FF', '#FFD1C1', '#E2C1FF', '#FFFAC1', '#C1FFF6']
            idx = abs(hash(subject)) % len(colors)
            return f'background-color: {colors[idx]}'
        st.dataframe(df.style.applymap(subject_color))
        # --- Download as CSV ---
        csv = df.to_csv().encode('utf-8')
        st.download_button("Download Timetable as CSV", data=csv, file_name="timetable.csv", mime="text/csv")
        # --- Visualization of free/busy slots for teachers ---
        st.subheader("Teacher Free/Busy Visualization (Whole Week)")
        for t in teacher_objs:
            t_slots = [f"{day} {slot[0].strftime('%H:%M')} - {slot[1].strftime('%H:%M')}" for day in DAYS for slot in CUSTOM_SLOTS]
            busy = set()
            for class_entries in timetable.values():
                for entry in class_entries:
                    if entry['teacher'] == t['name']:
                        busy.add(f"{entry['slot'][0]} {entry['slot'][1][0].strftime('%H:%M')} - {entry['slot'][1][1].strftime('%H:%M')}")
            status = ["🟩 Free" if slot not in busy else "🟥 Busy" for slot in t_slots]
            vis_df = pd.DataFrame({"Slot": t_slots, "Status": status})
            st.markdown(f"**{t['name']}**")
            st.dataframe(vis_df)
        # --- Visualize timetable for each class using the default table structure (with breaks) ---
        st.subheader("Class Timetables (Default Table Structure, With Breaks)")
        # Helper: map slot (day, period) to subject-teacher for each class
        for class_idx, class_name in enumerate(class_names):
            class_entries = timetable[class_name]
            slot_to_entry = {(entry['slot'][0], entry['slot'][1]): f"{entry['subject']}<br><small>{entry['teacher']}</small>" for entry in class_entries}
            # Build HTML table for this class
            html = "<style>td,th {text-align:center;vertical-align:middle;} .break-col {font-weight:bold;writing-mode:vertical-rl;text-orientation: mixed;} .break-header {white-space:pre-line;}</style>"
            html += "<table border='1' style='border-collapse:collapse;width:100%'>"
            # Header
            html += "<tr><th>Day/Time</th>"
            for col in all_columns:
                if any(col.startswith(b[1]) for b in BREAK_COLUMNS):
                    html += f"<th class='break-header'>{col}</th>"
                else:
                    html += f"<th>{col}</th>"
            html += "</tr>"
            for day in DAYS:
                html += f"<tr><td><b>{day}</b></td>"
                period_idx = 0
                for col in all_columns:
                    if col.startswith("Tea Break"):
                        html += f"<td class='break-col' rowspan='{len(DAYS)}'><div style='height:120px;display:flex;align-items:center;justify-content:center;'><span style='font-size:1.2em;writing-mode:vertical-rl;text-orientation:mixed;'>Tea Break</span></div></td>" if day == DAYS[0] else ""
                    elif col.startswith("Lunch Break"):
                        html += f"<td class='break-col' rowspan='{len(DAYS)}'><div style='height:120px;display:flex;align-items:center;justify-content:center;'><span style='font-size:1.2em;writing-mode:vertical-rl;text-orientation:mixed;'>Lunch Break</span></div></td>" if day == DAYS[0] else ""
                    else:
                        # This is a period column
                        if period_idx < len(CUSTOM_SLOTS):
                            slot_val = slot_to_entry.get((day, CUSTOM_SLOTS[period_idx]), "")
                            html += f"<td>{slot_val}</td>"
                        else:
                            html += "<td></td>"
                        period_idx += 1
                html += "</tr>"
            html += "</table>"
            if class_idx == 0:
                st.markdown(f"**{class_name}**")
            else:
                st.markdown(f"**{class_name}**")
            st.markdown(html, unsafe_allow_html=True)
            # Add download button for this class's timetable
            class_df = pd.DataFrame(index=DAYS, columns=all_columns)
            for day in DAYS:
                period_idx = 0
                for col in all_columns:
                    if any(col.startswith(b[1]) for b in BREAK_COLUMNS):
                        # This is a break column
                        class_df.loc[day, col] = col.split('\n')[0].strip() # Just the break name
                    else:
                        # This is a period column
                        if period_idx < len(CUSTOM_SLOTS):
                            slot_val = slot_to_entry.get((day, CUSTOM_SLOTS[period_idx]), "")
                            class_df.loc[day, col] = slot_val.replace('<br><small>', ' (').replace('</small>', ')')
                        else:
                            class_df.loc[day, col] = ""
                        period_idx += 1
            csv_for_class = class_df.to_csv().encode('utf-8')
            st.download_button(
                label=f"Download Timetable for {class_name}",
                data=csv_for_class,
                file_name=f"timetable_{class_name.replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_class_timetable_{class_idx}"
            )

if not st.session_state.timetable_generated:
    # Default Timetable Example (Blank Template with Breaks as Columns, 12hr format, merged cells, normal background, show break time)
    def to_12hr(t):
        return t.strftime('%I:%M %p').lstrip('0').replace(' 0', ' ')

    # Custom slot labels in 12hr format
    SLOT_LABELS_12HR = [f"{to_12hr(slot[0])} - {to_12hr(slot[1])}" for slot in CUSTOM_SLOTS]
    # Insert tea break and lunch break as columns at correct positions
    BREAK_COLUMNS = [
        ("10:00 AM - 10:30 AM", "Tea Break", 2),  # after 9:00-10:00
        ("12:30 PM - 2:00 PM", "Lunch Break", 4), # after 11:30-12:30
    ]
    all_columns = SLOT_LABELS_12HR.copy()
    for break_time, break_name, idx in sorted(BREAK_COLUMNS, key=lambda x: -x[2]):
        all_columns.insert(idx, f"{break_name}\n({break_time})")

    # Build HTML table for merged break columns
    html = "<style>td,th {text-align:center;vertical-align:middle;} .break-col {font-weight:bold;writing-mode:vertical-rl;text-orientation: mixed;} .break-header {white-space:pre-line;}</style>"
    html += "<table border='1' style='border-collapse:collapse;width:100%'>"
    # Header
    html += "<tr><th>Day/Time</th>"
    for col in all_columns:
        if any(col.startswith(b[1]) for b in BREAK_COLUMNS):
            # Show break name and time in header, normal background
            html += f"<th class='break-header'>{col}</th>"
        else:
            html += f"<th>{col}</th>"
    html += "</tr>"
    for day in DAYS:
        html += f"<tr><td><b>{day}</b></td>"
        for col in all_columns:
            if col.startswith("Tea Break"):
                html += f"<td class='break-col' rowspan='{len(DAYS)}'><div style='height:120px;display:flex;align-items:center;justify-content:center;'><span style='font-size:1.2em;writing-mode:vertical-rl;text-orientation:mixed;'>Tea Break</span></div></td>" if day == DAYS[0] else ""
            elif col.startswith("Lunch Break"):
                html += f"<td class='break-col' rowspan='{len(DAYS)}'><div style='height:120px;display:flex;align-items:center;justify-content:center;'><span style='font-size:1.2em;writing-mode:vertical-rl;text-orientation:mixed;'>Lunch Break</span></div></td>" if day == DAYS[0] else ""
            else:
                html += "<td></td>"
        html += "</tr>"
    html += "</table>"
    st.subheader("Default Timetable Template (Mon-Sat, Custom Slots, Breaks as Columns, 12hr Format)")
    st.markdown(html, unsafe_allow_html=True) 