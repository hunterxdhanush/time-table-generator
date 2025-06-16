from datetime import time, timedelta, datetime
from typing import List, Dict, Any, Tuple, Optional
import copy
import math

# Helper: generate time slots within a range (e.g., 1-hour slots)
def generate_time_slots(start: time, end: time, slot_minutes: int = 60) -> List[Tuple[time, time]]:
    slots = []
    current = datetime.combine(datetime.today(), start)
    end_dt = datetime.combine(datetime.today(), end)
    while current + timedelta(minutes=slot_minutes) <= end_dt:
        slot_start = current.time()
        slot_end = (current + timedelta(minutes=slot_minutes)).time()
        slots.append((slot_start, slot_end))
        current += timedelta(minutes=slot_minutes)
    return slots

# CSP Solver
class TimetableCSP:
    def __init__(self, classes: List[str], teachers: List[Dict[str, Any]], subjects: List[List[Dict[str, str]]], slot_minutes: int = 60, custom_slots: Optional[List[Tuple[time, time]]] = None, days: Optional[List[str]] = None):
        self.classes = classes
        self.teachers = teachers
        self.subjects = subjects  # subjects[class][subject-period] = {subject, teacher}
        self.slot_minutes = slot_minutes
        self.custom_slots = custom_slots
        self.days = days if days is not None else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        # Build all (day, slot) pairs for the week
        if custom_slots is not None:
            self.all_slots = [(day, slot) for day in self.days for slot in custom_slots]
            self.teacher_times = {
                t['name']: [
                    (day, slot)
                    for day in self.days
                    for slot in custom_slots
                    if t['start_time'] <= slot[0] and slot[1] <= t['end_time']
                ]
                for t in teachers
            }
        else:
            # Not used in app, but keep for completeness
            self.all_slots = None
            self.teacher_times = {
                t['name']: [
                    (day, slot)
                    for day in self.days
                    for slot in generate_time_slots(t['start_time'], t['end_time'], slot_minutes)
                ]
                for t in teachers
            }
        self.teacher_subjects = {t['name']: set(t.get('subjects', [])) for t in teachers}
        self.timetable = None

    def solve(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        # Prepare variables: for each class, each subject-period needs a slot
        variables = []
        for class_idx, class_subjects in enumerate(self.subjects):
            for subj in class_subjects:
                variables.append({
                    'class': self.classes[class_idx],
                    'subject': subj['name'],
                    'teacher': subj['teacher']
                })
        # Domains: for each variable, possible (day, slot) pairs (teacher's available slots)
        domains = []
        for var in variables:
            teacher = var['teacher']
            # Constraint 3: Only assign if teacher can teach the subject
            if teacher is None or var['subject'] not in self.teacher_subjects.get(teacher, set()):
                domains.append([])  # No valid slot, will fail
            else:
                domains.append(copy.deepcopy(self.teacher_times[teacher]))
        assignment = [None] * len(variables)
        if self.backtrack(variables, domains, assignment, 0):
            # Build timetable per class
            timetable = {c: [] for c in self.classes}
            for idx, var in enumerate(variables):
                entry = {
                    'subject': var['subject'],
                    'teacher': var['teacher'],
                    'slot': assignment[idx]  # (day, slot)
                }
                timetable[var['class']].append(entry)
            self.timetable = timetable
            return timetable
        return None

    def backtrack(self, variables, domains, assignment, idx) -> bool:
        if idx == len(variables):
            return True
        var = variables[idx]
        for slot in domains[idx]:
            if self.is_consistent(var, slot, variables, assignment, idx):
                assignment[idx] = slot
                if self.backtrack(variables, domains, assignment, idx + 1):
                    return True
                assignment[idx] = None
        return False

    def is_consistent(self, var, slot, variables, assignment, idx) -> bool:
        # slot is (day, period)
        for j in range(idx):
            if assignment[j] is None:
                continue
            # Same teacher, same (day, period)
            if var['teacher'] == variables[j]['teacher'] and slot == assignment[j]:
                return False
            # Same class, same (day, period)
            if var['class'] == variables[j]['class'] and slot == assignment[j]:
                return False
        # Constraint 2: Teacher must only be scheduled within their working hours (already handled by domain construction)
        # Constraint 3: Teacher must be able to teach the subject (already handled by domain construction)
        # Constraint 4: No same subject in adjacent periods for a class on the same day
        if self.custom_slots is not None:
            # slot = (day, period)
            day, period = slot
            for j in range(idx):
                if assignment[j] is None:
                    continue
                if var['class'] == variables[j]['class']:
                    other_day, other_period = assignment[j]
                    # Check if on the same day and adjacent periods
                    if day == other_day:
                        period_idx = self.custom_slots.index(period)
                        other_period_idx = self.custom_slots.index(other_period)
                        if abs(period_idx - other_period_idx) == 1:
                            if var['subject'] == variables[j]['subject']:
                                return False
        # Constraint 5: Limit subjects per day for a given class (for even distribution)
        # This constraint prevents too many sessions of the same subject for a class on a single day.
        if self.custom_slots is not None:
            day, current_period = slot
            subject_counts_today = {}
            for j in range(idx):
                if assignment[j] is None:
                    continue
                assigned_day, assigned_period = assignment[j]
                if var['class'] == variables[j]['class'] and day == assigned_day:
                    subj = variables[j]['subject']
                    subject_counts_today[subj] = subject_counts_today.get(subj, 0) + 1
            
            # Calculate the maximum allowed occurrences for the current subject on this specific day.
            # This is a heuristic to encourage even distribution across the week.
            total_periods_for_current_subject = 0
            # Find the total periods for this subject in the current class definition.
            for class_subj_list in self.subjects:
                for s in class_subj_list:
                    # Match by both subject name and teacher to be precise, as subjects can be taught by different teachers.
                    if s['name'] == var['subject'] and s['teacher'] == var['teacher']:
                        total_periods_for_current_subject = int(s['periods'])
                        break
                if total_periods_for_current_subject > 0:
                    break

            # Apply the constraint only if the subject has a defined number of periods.
            if total_periods_for_current_subject > 0:
                # Use math.ceil to ensure that if a subject has, for example, 5 periods over 6 days,
                # it can still be scheduled at least once on some days, e.g., 1 per day.
                max_per_day = math.ceil(total_periods_for_current_subject / len(self.days))
                if subject_counts_today.get(var['subject'], 0) >= max_per_day:
                    return False

        return True

# Test updated for week-long slots

def test_timetable_csp():
    from datetime import time
    print("Running Timetable CSP Test...")
    # Dummy teachers
    teachers = [
        {"name": "Alice", "start_time": time(9, 0), "end_time": time(13, 0), "subjects": ["Math", "Physics"]},
        {"name": "Bob", "start_time": time(10, 0), "end_time": time(16, 0), "subjects": ["Chemistry"]},
        {"name": "Carol", "start_time": time(8, 0), "end_time": time(12, 0), "subjects": ["Biology", "Math"]},
        {"name": "Dave", "start_time": time(11, 0), "end_time": time(15, 0), "subjects": ["Physics", "Chemistry"]},
    ]
    # Dummy classes
    classes = ["Class A", "Class B", "Class C"]
    # Subjects for each class (subject, teacher assigned by name)
    subjects = [
        [
            {"subject": "Math", "teacher": "Alice"},
            {"subject": "Chemistry", "teacher": "Bob"},
        ],
        [
            {"subject": "Physics", "teacher": "Dave"},
            {"subject": "Math", "teacher": "Carol"},
        ],
        [
            {"subject": "Biology", "teacher": "Carol"},
            {"subject": "Physics", "teacher": "Alice"},
        ],
    ]
    custom_slots = [
        (time(8, 0), time(9, 0)),
        (time(9, 0), time(10, 0)),
        (time(10, 30), time(11, 30)),
        (time(11, 30), time(12, 30)),
        (time(14, 0), time(15, 0)),
        (time(15, 0), time(16, 0)),
        (time(16, 0), time(17, 0)),
    ]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    csp = TimetableCSP(classes, teachers, subjects, custom_slots=custom_slots, days=days)
    timetable = csp.solve()
    assert timetable is not None, "No valid timetable generated!"
    # Check constraints
    slot_teacher = {}
    for class_name, entries in timetable.items():
        for entry in entries:
            slot = entry['slot']  # (day, period)
            teacher = entry['teacher']
            subject = entry['subject']
            # 1. No teacher double-booked
            if (teacher, slot) in slot_teacher:
                raise AssertionError(f"Teacher {teacher} double-booked at {slot}!")
            slot_teacher[(teacher, slot)] = (class_name, subject)
            # 2. Teacher only within working hours (already ensured by slot generation)
            t = next(t for t in teachers if t['name'] == teacher)
            period = slot[1]
            assert t['start_time'] <= period[0] and period[1] <= t['end_time'], f"Teacher {teacher} scheduled outside working hours!"
            # 3. Teacher can teach subject
            assert subject in t['subjects'], f"Teacher {teacher} cannot teach {subject}!"
    print("All constraints respected. Generated Timetable:")
    for class_name, entries in timetable.items():
        print(f"{class_name}:")
        for entry in entries:
            day, period = entry['slot']
            print(f"  {day} {period[0].strftime('%H:%M')} - {period[1].strftime('%H:%M')}: {entry['subject']} - {entry['teacher']}")
    print("Test passed!")

if __name__ == "__main__":
    test_timetable_csp() 