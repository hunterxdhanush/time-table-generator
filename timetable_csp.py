from datetime import time, timedelta, datetime
from typing import List, Dict, Any, Tuple, Optional
import copy

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
    def __init__(self, classes: List[str], teachers: List[Dict[str, Any]], subjects: List[List[Dict[str, str]]], slot_minutes: int = 60):
        self.classes = classes
        self.teachers = teachers
        self.subjects = subjects  # subjects[class][subject-period] = {subject, teacher}
        self.slot_minutes = slot_minutes
        self.teacher_times = {t['name']: generate_time_slots(t['start_time'], t['end_time'], slot_minutes) for t in teachers}
        self.teacher_subjects = {t['name']: set(t.get('subjects', [])) for t in teachers}
        self.timetable = None

    def solve(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        # Prepare variables: for each class, each subject-period needs a slot
        variables = []
        for class_idx, class_subjects in enumerate(self.subjects):
            for subj in class_subjects:
                variables.append({
                    'class': self.classes[class_idx],
                    'subject': subj['subject'],
                    'teacher': subj['teacher']
                })
        # Domains: for each variable, possible time slots (teacher's available slots)
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
                    'slot': assignment[idx]
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
        # Constraint 1: No teacher double-booked, no class double-booked at the same time
        for j in range(idx):
            if assignment[j] is None:
                continue
            # Same teacher, same slot
            if var['teacher'] == variables[j]['teacher'] and slot == assignment[j]:
                return False
            # Same class, same slot
            if var['class'] == variables[j]['class'] and slot == assignment[j]:
                return False
        # Constraint 2: Teacher must only be scheduled within their working hours (already handled by domain construction)
        # Constraint 3: Teacher must be able to teach the subject (already handled by domain construction)
        return True

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
    csp = TimetableCSP(classes, teachers, subjects)
    timetable = csp.solve()
    assert timetable is not None, "No valid timetable generated!"
    # Check constraints
    slot_teacher = {}
    for class_name, entries in timetable.items():
        for entry in entries:
            slot = entry['slot']
            teacher = entry['teacher']
            subject = entry['subject']
            # 1. No teacher double-booked
            if (teacher, slot) in slot_teacher:
                raise AssertionError(f"Teacher {teacher} double-booked at {slot}!")
            slot_teacher[(teacher, slot)] = (class_name, subject)
            # 2. Teacher only within working hours (already ensured by slot generation)
            t = next(t for t in teachers if t['name'] == teacher)
            assert t['start_time'] <= slot[0] and slot[1] <= t['end_time'], f"Teacher {teacher} scheduled outside working hours!"
            # 3. Teacher can teach subject
            assert subject in t['subjects'], f"Teacher {teacher} cannot teach {subject}!"
    print("All constraints respected. Generated Timetable:")
    for class_name, entries in timetable.items():
        print(f"{class_name}:")
        for entry in entries:
            print(f"  {entry['slot'][0].strftime('%H:%M')} - {entry['slot'][1].strftime('%H:%M')}: {entry['subject']} - {entry['teacher']}")
    print("Test passed!")

if __name__ == "__main__":
    test_timetable_csp() 