from unittest.mock import MagicMock, patch
import threading
import curses

# Mock curses functions and constants for testing
curses.initscr = MagicMock()
curses.endwin = MagicMock()
curses.curs_set = MagicMock()
curses.noecho = MagicMock()
curses.cbreak = MagicMock()
curses.start_color = MagicMock()
curses.use_default_colors = MagicMock()
curses.init_pair = MagicMock()
curses.color_pair = MagicMock(return_value=0)
curses.A_BOLD = 1
curses.A_REVERSE = 2
curses.A_DIM = 4
curses.ACS_BLOCK = 'X'
curses.KEY_UP = 259
curses.KEY_DOWN = 258
curses.KEY_LEFT = 260
curses.KEY_RIGHT = 261
curses.KEY_HOME = 262
curses.KEY_END = 360
curses.KEY_ENTER = 10
curses.KEY_BACKSPACE = 263
curses.KEY_DC = 330
curses.KEY_PPAGE = 339
curses.KEY_NPAGE = 338
curses.is_term_resized = MagicMock(return_value=False)
curses.update_lines_cols = MagicMock()
curses.wrapper = lambda func, *args, **kwargs: func(curses.initscr(), *args, **kwargs)


class MockApp:
    """Mock App class for testing curses screens."""

    def __init__(self):
        self.credentials_manager = MagicMock()
        self.content_fetcher = MagicMock()
        self.dataset_creator = MagicMock()
        self.dataset_manager = MagicMock()
        self.push_screen = MagicMock()
        self.pop_screen = MagicMock()
        self.exit = MagicMock()
        self.add_background_task = MagicMock()
        self.stdscr = MagicMock()
        self.stdscr.getmaxyx.return_value = (24, 80)
        self.running = True
        self.screen_stack = []


def setup_mock_ui_test():
    """Setup common mocks for UI testing."""
    # Create a mock curses screen
    mock_stdscr = MagicMock()
    mock_stdscr.getmaxyx.return_value = (24, 80)
    
    # Create mock progress bar and log container
    mock_progress_bar = MagicMock()
    mock_progress_bar.update = MagicMock()
    
    mock_log_container = MagicMock()
    mock_log_container.add_message = MagicMock()
    
    # Create mock screen with the necessary attributes
    mock_screen = MagicMock()
    mock_screen.stdscr = mock_stdscr
    mock_screen.progress_bar = mock_progress_bar
    mock_screen.log_view = mock_log_container
    mock_screen._last_logged_progress = 0
    mock_screen.status_message = MagicMock()
    mock_screen.status_message.set_text = MagicMock()

    return mock_screen, mock_progress_bar, mock_log_container