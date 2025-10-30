"""
Implements a runtime data model that can be used to access
the properties contained in the issues JSON.
"""

from typing import List
from enum import Enum
from datetime import datetime, timedelta
from dateutil import parser


class State(str, Enum):
    # Whether issue is open or closed
    open = 'open'
    closed = 'closed'


class Event:
    def __init__(self, jobj: any):
        self.event_type: str = None
        self.author: str = None
        self.event_date: datetime = None
        self.label: str = None
        self.comment: str = None

        if jobj is not None:
            self.from_json(jobj)

    def from_json(self, jobj: any):
        self.event_type = jobj.get('event_type')
        self.author = jobj.get('author')
        try:
            self.event_date = parser.parse(jobj.get('event_date'))
        except Exception:
            pass
        self.label = jobj.get('label')
        self.comment = jobj.get('comment')

    def is_close_event(self) -> bool:
        return self.event_type == 'closed'

    def is_comment_event(self) -> bool:
        return self.event_type == 'commented'


class Issue:
    def __init__(self, jobj: any = None):
        self.url: str = None
        self.creator: str = None
        self.labels: List[str] = []
        self.state: State = None
        self.assignees: List[str] = []
        self.title: str = None
        self.text: str = None
        self.number: int = -1
        self.created_date: datetime = None
        self.updated_date: datetime = None
        self.timeline_url: str = None
        self.events: List[Event] = []
        self.assigned_priority: str = None

        if jobj is not None:
            self.from_json(jobj)

    def from_json(self, jobj: any):
        self.url = jobj.get('url')
        self.creator = jobj.get('creator')
        self.labels = jobj.get('labels', [])
        self.state = State[jobj.get('state')]
        self.assignees = jobj.get('assignees', [])
        self.title = jobj.get('title')
        self.text = jobj.get('text')
        try:
            self.number = int(jobj.get('number', '-1'))
        except Exception:
            pass
        try:
            self.created_date = parser.parse(jobj.get('created_date'))
        except Exception:
            pass
        try:
            self.updated_date = parser.parse(jobj.get('updated_date'))
        except Exception:
            pass
        self.timeline_url = jobj.get('timeline_url')
        self.events = [Event(jevent) for jevent in jobj.get('events', [])]


    def get_labels(self) -> List[str]:
        return self.labels

    def get_creation_date(self) -> datetime:
        return self.created_date

    def get_closure_date(self) -> datetime:
        closes = [e.event_date for e in self.events if e.is_close_event()]
        return min(closes) if closes else None

    def is_closed(self) -> bool:
        return self.state == State.closed

    def get_resolution_time(self) -> timedelta:
        if self.created_date and self.get_closure_date():
            return self.get_closure_date() - self.created_date
        return None

    def get_comment_count(self) -> int:
        return sum(1 for e in self.events if e.is_comment_event())

    def get_event_count(self) -> int:
        return len(self.events)

    def get_text_content(self) -> str:
        return self.text

    def get_title(self) -> str:
        return self.title

    def set_priority(self, priority: str):
        self.assigned_priority = priority

    def get_priority(self) -> str:
        return self.assigned_priority


class Contributor:
    def __init__(self, username: str):
        self.username = username
        self.issues_created: List[Issue] = []
        self.issues_closed: List[Issue] = []
        self.comments: List[Event] = []
        self.first_activity: datetime = None
        self.last_activity: datetime = None
        
    def add_issue(self, issue: Issue):
        self.issues_created.append(issue)
        self._update_activity(issue.created_date)

    def add_comment(self, event: Event):
        self.comments.append(event)
        self._update_activity(event.event_date)

    def _update_activity(self, date: datetime):
        if not date:
            return
        if not self.first_activity or date < self.first_activity:
            self.first_activity = date
        if not self.last_activity or date > self.last_activity:
            self.last_activity = date
    
    def add_closed_issue(self, issue: Issue):
        closure_date = issue.get_closure_date()
        if closure_date:
            self.issues_closed.append(issue)
            self._update_activity(closure_date)

    def get_activity_count(self) -> int:
        return (
            len(self.issues_created)
            + len(self.issues_closed)
            + len(self.comments)
        )

    def get_activity_count_by_year(self, year: int) -> int:
        # Returns this contributor's activity count for a specific year
        # Activity = issues created + comments + issues closed in that year
        count = 0
        # Issues created
        for issue in self.issues_created:
            if issue.created_date and issue.created_date.year == year:
                count += 1
        # Comments
        for comment in self.comments:
            if comment.event_date and comment.event_date.year == year:
                count += 1
        # Issues closed
        for issue in getattr(self, "issues_closed", []):
            closure_date = issue.get_closure_date()
            if closure_date and closure_date.year == year:
                count += 1

        return count

    def get_active_years(self) -> set[int]:
        
        # Returns all years in which this contributor had any activity.
        years = set()
        years.update([i.created_date.year for i in self.issues_created if i.created_date])
        years.update([e.event_date.year for e in self.comments if e.event_date])
        years.update([
            cd.year for i in getattr(self, "issues_closed", [])
            for cd in [i.get_closure_date()] if cd
        ])
        return years

class Comment:
    def __init__(self, id: str, event_date=None, issue_id=None):
        self.id = id
        self.event_date = event_date
        self.issue_id = issue_id

    def __repr__(self):
        return f"<Comment id={self.id} date={self.event_date}>"