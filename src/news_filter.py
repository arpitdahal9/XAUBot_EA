"""
News Filter for Backtest Avoidance

Implements news event filtering ONLY for backtesting.
Live trading will be manually paused during news.

Major USD Events to Avoid:
- CPI (Consumer Price Index)
- NFP (Non-Farm Payrolls)
- FOMC (Federal Open Market Committee)

Avoidance Window: ±30 minutes around event time
"""

import os
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class NewsEvent:
    """A news event to avoid."""
    timestamp: datetime
    event_type: str  # CPI, NFP, FOMC
    currency: str  # USD, EUR, etc.
    importance: str  # HIGH, MEDIUM
    description: str = ""


class NewsFilter:
    """
    Filter trades around major news events.
    
    ONLY used in backtesting - live trading pauses manually.
    """
    
    # Known major USD news events (2020-2026 sample)
    # Format: (month, day, hour_utc)
    FOMC_DATES = [
        # 2020
        (2020, 1, 29, 19), (2020, 3, 3, 19), (2020, 3, 15, 17), (2020, 4, 29, 18),
        (2020, 6, 10, 18), (2020, 7, 29, 18), (2020, 9, 16, 18), (2020, 11, 5, 19),
        (2020, 12, 16, 19),
        # 2021
        (2021, 1, 27, 19), (2021, 3, 17, 18), (2021, 4, 28, 18), (2021, 6, 16, 18),
        (2021, 7, 28, 18), (2021, 9, 22, 18), (2021, 11, 3, 18), (2021, 12, 15, 19),
        # 2022
        (2022, 1, 26, 19), (2022, 3, 16, 18), (2022, 5, 4, 18), (2022, 6, 15, 18),
        (2022, 7, 27, 18), (2022, 9, 21, 18), (2022, 11, 2, 18), (2022, 12, 14, 19),
        # 2023
        (2023, 2, 1, 19), (2023, 3, 22, 18), (2023, 5, 3, 18), (2023, 6, 14, 18),
        (2023, 7, 26, 18), (2023, 9, 20, 18), (2023, 11, 1, 18), (2023, 12, 13, 19),
        # 2024
        (2024, 1, 31, 19), (2024, 3, 20, 18), (2024, 5, 1, 18), (2024, 6, 12, 18),
        (2024, 7, 31, 18), (2024, 9, 18, 18), (2024, 11, 7, 19), (2024, 12, 18, 19),
        # 2025
        (2025, 1, 29, 19), (2025, 3, 19, 18), (2025, 5, 7, 18), (2025, 6, 18, 18),
        (2025, 7, 30, 18), (2025, 9, 17, 18), (2025, 11, 5, 19), (2025, 12, 17, 19),
        # 2026
        (2026, 1, 28, 19), (2026, 3, 18, 18), (2026, 5, 6, 18),
    ]
    
    # NFP - First Friday of each month at 8:30 EST (13:30 UTC)
    # CPI - Around 12th-15th of each month at 8:30 EST (13:30 UTC)
    
    def __init__(self, avoidance_minutes: int = 30):
        """
        Initialize News Filter.
        
        Args:
            avoidance_minutes: Minutes before/after event to avoid
        """
        self.logger = logging.getLogger(__name__)
        self.avoidance_minutes = avoidance_minutes
        
        self._events: List[NewsEvent] = []
        self._event_set: Set[datetime] = set()  # For fast lookup
        
        # Build event list
        self._build_event_list()
        
        self.logger.info(f"[NEWS_FILTER] Loaded {len(self._events)} events")
        self.logger.info(f"[NEWS_FILTER] Avoidance window: ±{self.avoidance_minutes} minutes")
    
    def _build_event_list(self) -> None:
        """Build list of all news events."""
        # Add FOMC events
        for year, month, day, hour in self.FOMC_DATES:
            try:
                dt = datetime(year, month, day, hour, 0)
                self._events.append(NewsEvent(
                    timestamp=dt,
                    event_type="FOMC",
                    currency="USD",
                    importance="HIGH",
                    description="FOMC Interest Rate Decision"
                ))
                self._event_set.add(dt)
            except ValueError:
                continue
        
        # Generate NFP dates (first Friday of each month)
        for year in range(2020, 2027):
            for month in range(1, 13):
                # Find first Friday
                day = 1
                dt = datetime(year, month, day, 13, 30)  # 8:30 EST = 13:30 UTC
                while dt.weekday() != 4:  # 4 = Friday
                    day += 1
                    dt = datetime(year, month, day, 13, 30)
                
                self._events.append(NewsEvent(
                    timestamp=dt,
                    event_type="NFP",
                    currency="USD",
                    importance="HIGH",
                    description="Non-Farm Payrolls"
                ))
                self._event_set.add(dt)
        
        # Generate CPI dates (usually around 13th of month)
        for year in range(2020, 2027):
            for month in range(1, 13):
                try:
                    # CPI typically released on 13th-15th at 8:30 EST
                    dt = datetime(year, month, 13, 13, 30)
                    # Skip weekends
                    while dt.weekday() >= 5:
                        dt += timedelta(days=1)
                    
                    self._events.append(NewsEvent(
                        timestamp=dt,
                        event_type="CPI",
                        currency="USD",
                        importance="HIGH",
                        description="Consumer Price Index"
                    ))
                    self._event_set.add(dt)
                except ValueError:
                    continue
        
        # Sort by timestamp
        self._events.sort(key=lambda x: x.timestamp)
    
    def should_skip_trade(self, timestamp: datetime) -> bool:
        """
        Check if trade should be skipped due to nearby news.
        
        Args:
            timestamp: Time of potential trade entry
            
        Returns:
            True if trade should be skipped
        """
        window = timedelta(minutes=self.avoidance_minutes)
        
        for event in self._events:
            if abs((timestamp - event.timestamp).total_seconds()) <= window.total_seconds():
                self.logger.debug(
                    f"[NEWS_FILTER] Skipping trade at {timestamp}: "
                    f"Near {event.event_type} at {event.timestamp}"
                )
                return True
        
        return False
    
    def get_next_event(self, timestamp: datetime) -> Optional[NewsEvent]:
        """Get the next news event after given timestamp."""
        for event in self._events:
            if event.timestamp > timestamp:
                return event
        return None
    
    def get_events_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[NewsEvent]:
        """Get all events in date range."""
        return [e for e in self._events if start <= e.timestamp <= end]
    
    def load_custom_calendar(self, csv_path: str) -> None:
        """
        Load additional events from CSV file.
        
        CSV Format: timestamp,event_type,currency,importance,description
        """
        if not os.path.exists(csv_path):
            self.logger.warning(f"[NEWS_FILTER] Calendar file not found: {csv_path}")
            return
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dt = datetime.fromisoformat(row['timestamp'])
                        event = NewsEvent(
                            timestamp=dt,
                            event_type=row.get('event_type', 'CUSTOM'),
                            currency=row.get('currency', 'USD'),
                            importance=row.get('importance', 'HIGH'),
                            description=row.get('description', '')
                        )
                        self._events.append(event)
                        self._event_set.add(dt)
                    except (ValueError, KeyError):
                        continue
            
            # Re-sort after adding
            self._events.sort(key=lambda x: x.timestamp)
            self.logger.info(f"[NEWS_FILTER] Loaded custom calendar: {csv_path}")
            
        except Exception as e:
            self.logger.error(f"[NEWS_FILTER] Failed to load calendar: {e}")
    
    def export_calendar(self, csv_path: str, start: datetime, end: datetime) -> None:
        """Export events in range to CSV for verification."""
        events = self.get_events_in_range(start, end)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'event_type', 'currency', 'importance', 'description'])
            
            for e in events:
                writer.writerow([
                    e.timestamp.isoformat(),
                    e.event_type,
                    e.currency,
                    e.importance,
                    e.description
                ])
        
        self.logger.info(f"[NEWS_FILTER] Exported {len(events)} events to {csv_path}")
