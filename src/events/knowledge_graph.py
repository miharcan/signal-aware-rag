class EventGraph:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        if event:
            self.events.append(event)

    def query_by_company(self, company):
        return [
            e for e in self.events
            if e.get("company") == company
        ]

    def query_by_event_type(self, event_type):
        return [
            e for e in self.events
            if e.get("event_type") == event_type
        ]