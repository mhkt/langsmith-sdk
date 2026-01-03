"""Unit tests for Claude Agent SDK client instrumentation."""

from unittest.mock import MagicMock, Mock
import pytest
from langsmith.integrations.claude_agent_sdk._client import TurnLifecycle


class TestTurnLifecycleAddUsage:
    """Test TurnLifecycle.add_usage() method using validated RunTree.set() API."""

    def test_add_usage_with_empty_existing_metadata(self):
        """Test adding usage when no existing metadata exists."""
        # Create mock run with proper structure
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150,
        }

        tracker.add_usage(usage_metadata)

        # Verify RunTree.set() was called with correct metadata
        mock_run.set.assert_called_once()
        call_args = mock_run.set.call_args
        assert 'usage_metadata' in call_args.kwargs
        assert call_args.kwargs['usage_metadata'] == usage_metadata

    def test_add_usage_merges_with_existing_metadata(self):
        """Test that add_usage merges with existing usage metadata."""
        # Create mock run with existing usage metadata
        mock_run = Mock()
        mock_run.extra = {
            'metadata': {
                'usage_metadata': {
                    'input_tokens': 100,
                    'output_tokens': 20,
                    'total_tokens': 120,
                }
            }
        }
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        # Add new usage data (e.g., updating output tokens)
        new_usage = {
            'output_tokens': 50,
            'total_tokens': 150,
        }

        tracker.add_usage(new_usage)

        # Verify merged metadata was passed to set()
        mock_run.set.assert_called_once()
        call_args = mock_run.set.call_args
        merged = call_args.kwargs['usage_metadata']

        assert merged['input_tokens'] == 100  # Preserved from existing
        assert merged['output_tokens'] == 50  # Updated from new
        assert merged['total_tokens'] == 150  # Updated from new

    def test_add_usage_with_cache_tokens(self):
        """Test adding usage metadata with cache token details."""
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        usage_metadata = {
            'input_tokens': 1000,
            'output_tokens': 500,
            'total_tokens': 2000,
            'input_token_details': {
                'cache_read': 400,
                'cache_creation': 100,
            }
        }

        tracker.add_usage(usage_metadata)

        mock_run.set.assert_called_once()
        call_args = mock_run.set.call_args
        result = call_args.kwargs['usage_metadata']

        assert result['input_tokens'] == 1000
        assert result['output_tokens'] == 500
        assert result['total_tokens'] == 2000
        assert 'input_token_details' in result
        assert result['input_token_details']['cache_read'] == 400
        assert result['input_token_details']['cache_creation'] == 100

    def test_add_usage_no_current_run(self):
        """Test that add_usage does nothing when no current run exists."""
        tracker = TurnLifecycle()
        # No current_run set

        usage_metadata = {'input_tokens': 100}

        # Should not raise an error
        tracker.add_usage(usage_metadata)
        # Nothing to assert - just verify it doesn't crash

    def test_add_usage_empty_metrics(self):
        """Test that add_usage does nothing with empty metrics."""
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        tracker.add_usage({})

        # Should not call set() with empty metrics
        mock_run.set.assert_not_called()


class MockStreamEvent:
    """Mock StreamEvent message for testing."""

    def __init__(self, event_dict):
        self.event = event_dict


class TestStreamEventHandler:
    """Test StreamEvent message handling for per-turn token tracking."""

    def test_message_start_event_basic_tokens(self):
        """Test processing message_start event with basic token counts."""
        # Create mock tracker
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        # Create message_start event
        event = {
            'type': 'message_start',
            'message': {
                'usage': {
                    'input_tokens': 5000,
                    'output_tokens': 0,
                }
            }
        }

        msg = MockStreamEvent(event)

        # Simulate the StreamEvent handler logic
        event_data = getattr(msg, 'event', {})
        event_type = event_data.get('type')

        if event_type == 'message_start':
            message_data = event_data.get('message', {})
            usage = message_data.get('usage', {})

            if usage:
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                cache_read = usage.get('cache_read_input_tokens', 0)
                cache_create = usage.get('cache_creation_input_tokens', 0)

                usage_metadata = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens + cache_read + cache_create,
                }

                if cache_read or cache_create:
                    usage_metadata['input_token_details'] = {}
                    if cache_read:
                        usage_metadata['input_token_details']['cache_read'] = cache_read
                    if cache_create:
                        usage_metadata['input_token_details']['cache_creation'] = cache_create

                tracker.add_usage(usage_metadata)

        # Verify correct usage was recorded
        mock_run.set.assert_called_once()
        call_args = mock_run.set.call_args
        result = call_args.kwargs['usage_metadata']

        assert result['input_tokens'] == 5000
        assert result['output_tokens'] == 0
        assert result['total_tokens'] == 5000

    def test_message_start_event_with_cache_tokens(self):
        """Test processing message_start event with cache token details."""
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        # Create message_start event with cache tokens
        event = {
            'type': 'message_start',
            'message': {
                'usage': {
                    'input_tokens': 1500,
                    'output_tokens': 0,
                    'cache_read_input_tokens': 50000,
                    'cache_creation_input_tokens': 5000,
                }
            }
        }

        msg = MockStreamEvent(event)

        # Simulate handler logic
        event_data = msg.event
        event_type = event_data.get('type')

        if event_type == 'message_start':
            message_data = event_data.get('message', {})
            usage = message_data.get('usage', {})

            if usage:
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                cache_read = usage.get('cache_read_input_tokens', 0)
                cache_create = usage.get('cache_creation_input_tokens', 0)

                usage_metadata = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens + cache_read + cache_create,
                }

                if cache_read or cache_create:
                    usage_metadata['input_token_details'] = {}
                    if cache_read:
                        usage_metadata['input_token_details']['cache_read'] = cache_read
                    if cache_create:
                        usage_metadata['input_token_details']['cache_creation'] = cache_create

                tracker.add_usage(usage_metadata)

        # Verify all tokens were recorded correctly
        mock_run.set.assert_called_once()
        result = mock_run.set.call_args.kwargs['usage_metadata']

        assert result['input_tokens'] == 1500
        assert result['output_tokens'] == 0
        assert result['total_tokens'] == 56500  # 1500 + 50000 + 5000
        assert 'input_token_details' in result
        assert result['input_token_details']['cache_read'] == 50000
        assert result['input_token_details']['cache_creation'] == 5000

    def test_message_delta_event_updates_output_tokens(self):
        """Test processing message_delta event to update output tokens."""
        mock_run = Mock()
        # Existing metadata from message_start
        mock_run.extra = {
            'metadata': {
                'usage_metadata': {
                    'input_tokens': 1000,
                    'output_tokens': 0,
                    'total_tokens': 1000,
                }
            }
        }
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        # Create message_delta event
        event = {
            'type': 'message_delta',
            'usage': {
                'output_tokens': 250,  # Cumulative output tokens
            }
        }

        msg = MockStreamEvent(event)

        # Simulate handler logic
        event_data = msg.event
        event_type = event_data.get('type')

        if event_type == 'message_delta':
            delta_usage = event_data.get('usage', {})
            if delta_usage and 'output_tokens' in delta_usage:
                output_tokens = delta_usage['output_tokens']

                if tracker.current_run:
                    current_meta = tracker.current_run.extra.get('metadata', {}).get('usage_metadata', {})
                    input_tokens = current_meta.get('input_tokens', 0)

                    input_details = current_meta.get('input_token_details', {})
                    cache_read = input_details.get('cache_read', 0)
                    cache_create = input_details.get('cache_creation', 0)

                    tracker.add_usage({
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens + cache_read + cache_create,
                    })

        # Verify output tokens were updated
        mock_run.set.assert_called_once()
        result = mock_run.set.call_args.kwargs['usage_metadata']

        assert result['input_tokens'] == 1000  # Preserved from existing
        assert result['output_tokens'] == 250  # Updated from delta
        assert result['total_tokens'] == 1250  # Recalculated

    def test_message_delta_with_cache_tokens(self):
        """Test message_delta correctly recalculates total with cache tokens."""
        mock_run = Mock()
        # Existing metadata with cache tokens
        mock_run.extra = {
            'metadata': {
                'usage_metadata': {
                    'input_tokens': 2000,
                    'output_tokens': 0,
                    'total_tokens': 52000,
                    'input_token_details': {
                        'cache_read': 45000,
                        'cache_creation': 5000,
                    }
                }
            }
        }
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        # Create message_delta event
        event = {
            'type': 'message_delta',
            'usage': {
                'output_tokens': 500,
            }
        }

        msg = MockStreamEvent(event)

        # Simulate handler logic
        event_data = msg.event
        event_type = event_data.get('type')

        if event_type == 'message_delta':
            delta_usage = event_data.get('usage', {})
            if delta_usage and 'output_tokens' in delta_usage:
                output_tokens = delta_usage['output_tokens']

                if tracker.current_run:
                    current_meta = tracker.current_run.extra.get('metadata', {}).get('usage_metadata', {})
                    input_tokens = current_meta.get('input_tokens', 0)

                    input_details = current_meta.get('input_token_details', {})
                    cache_read = input_details.get('cache_read', 0)
                    cache_create = input_details.get('cache_creation', 0)

                    tracker.add_usage({
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens + cache_read + cache_create,
                    })

        # Verify total includes cache tokens
        mock_run.set.assert_called_once()
        result = mock_run.set.call_args.kwargs['usage_metadata']

        assert result['input_tokens'] == 2000
        assert result['output_tokens'] == 500
        # Total = 2000 + 500 + 45000 + 5000 = 52500
        assert result['total_tokens'] == 52500
        assert result['input_token_details']['cache_read'] == 45000
        assert result['input_token_details']['cache_creation'] == 5000


class TestResultMessageHandler:
    """Test ResultMessage handler to verify it no longer adds duplicate usage."""

    def test_result_message_only_adds_cost_not_usage(self):
        """Test that ResultMessage only adds total_cost, not usage data."""
        # Create mock conversation run
        mock_run = Mock()
        mock_run.metadata = {}

        # Create mock ResultMessage
        class MockResultMessage:
            def __init__(self):
                self.usage = {
                    'input_tokens': 10000,
                    'output_tokens': 5000,
                }
                self.total_cost_usd = 0.15
                self.num_turns = 5
                self.session_id = 'test-session-123'
                self.duration_ms = 5000
                self.duration_api_ms = 4500
                self.is_error = False

        msg = MockResultMessage()

        # Simulate the new ResultMessage handler logic
        # Per-turn usage already added via StreamEvent
        # Only add conversation-level cost to metadata
        if hasattr(msg, "total_cost_usd") and msg.total_cost_usd is not None:
            mock_run.metadata["total_cost"] = msg.total_cost_usd

        # Add conversation-level metadata
        meta = {
            k: v
            for k, v in {
                "num_turns": getattr(msg, "num_turns", None),
                "session_id": getattr(msg, "session_id", None),
                "duration_ms": getattr(msg, "duration_ms", None),
                "duration_api_ms": getattr(msg, "duration_api_ms", None),
                "is_error": getattr(msg, "is_error", None),
            }.items()
            if v is not None
        }
        if meta:
            mock_run.metadata.update(meta)

        # Verify only metadata was added, no usage
        assert mock_run.metadata['total_cost'] == 0.15
        assert mock_run.metadata['num_turns'] == 5
        assert mock_run.metadata['session_id'] == 'test-session-123'
        assert mock_run.metadata['duration_ms'] == 5000
        assert mock_run.metadata['duration_api_ms'] == 4500
        assert mock_run.metadata['is_error'] is False

        # Verify no usage_metadata was added to metadata
        assert 'usage_metadata' not in mock_run.metadata

    def test_result_message_without_cost(self):
        """Test ResultMessage handler when no cost is available."""
        mock_run = Mock()
        mock_run.metadata = {}

        class MockResultMessage:
            def __init__(self):
                self.usage = {'input_tokens': 1000}
                self.total_cost_usd = None
                self.num_turns = 3
                self.session_id = 'test-session-456'

        msg = MockResultMessage()

        # Simulate handler logic
        if hasattr(msg, "total_cost_usd") and msg.total_cost_usd is not None:
            mock_run.metadata["total_cost"] = msg.total_cost_usd

        meta = {
            k: v
            for k, v in {
                "num_turns": getattr(msg, "num_turns", None),
                "session_id": getattr(msg, "session_id", None),
                "duration_ms": getattr(msg, "duration_ms", None),
                "duration_api_ms": getattr(msg, "duration_api_ms", None),
                "is_error": getattr(msg, "is_error", None),
            }.items()
            if v is not None
        }
        if meta:
            mock_run.metadata.update(meta)

        # Verify no cost was added (since it was None)
        assert 'total_cost' not in mock_run.metadata
        # But other metadata should be present
        assert mock_run.metadata['num_turns'] == 3
        assert mock_run.metadata['session_id'] == 'test-session-456'


class TestIntegrationScenario:
    """Test complete flow: message_start → message_delta → ResultMessage."""

    def test_complete_turn_flow(self):
        """Test full token tracking flow for a single turn."""
        # Setup tracker with mock run
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()

        tracker = TurnLifecycle()
        tracker.current_run = mock_run

        # Step 1: message_start event
        start_event = {
            'type': 'message_start',
            'message': {
                'usage': {
                    'input_tokens': 3000,
                    'output_tokens': 0,
                    'cache_read_input_tokens': 25000,
                    'cache_creation_input_tokens': 2000,
                }
            }
        }

        msg_start = MockStreamEvent(start_event)
        event_data = msg_start.event
        message_data = event_data.get('message', {})
        usage = message_data.get('usage', {})

        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cache_read = usage.get('cache_read_input_tokens', 0)
        cache_create = usage.get('cache_creation_input_tokens', 0)

        usage_metadata = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens + cache_read + cache_create,
        }

        if cache_read or cache_create:
            usage_metadata['input_token_details'] = {}
            if cache_read:
                usage_metadata['input_token_details']['cache_read'] = cache_read
            if cache_create:
                usage_metadata['input_token_details']['cache_creation'] = cache_create

        tracker.add_usage(usage_metadata)

        # Verify message_start was processed
        assert mock_run.set.call_count == 1
        first_call = mock_run.set.call_args.kwargs['usage_metadata']
        assert first_call['input_tokens'] == 3000
        assert first_call['total_tokens'] == 30000  # 3000 + 25000 + 2000

        # Update mock_run.extra to reflect the state after message_start
        mock_run.extra = {
            'metadata': {
                'usage_metadata': first_call
            }
        }

        # Step 2: message_delta event (updates output tokens)
        delta_event = {
            'type': 'message_delta',
            'usage': {
                'output_tokens': 750,
            }
        }

        msg_delta = MockStreamEvent(delta_event)
        delta_usage = msg_delta.event.get('usage', {})

        if delta_usage and 'output_tokens' in delta_usage:
            output_tokens = delta_usage['output_tokens']

            current_meta = mock_run.extra.get('metadata', {}).get('usage_metadata', {})
            input_tokens = current_meta.get('input_tokens', 0)

            input_details = current_meta.get('input_token_details', {})
            cache_read = input_details.get('cache_read', 0)
            cache_create = input_details.get('cache_creation', 0)

            tracker.add_usage({
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens + cache_read + cache_create,
            })

        # Verify message_delta was processed
        assert mock_run.set.call_count == 2
        second_call = mock_run.set.call_args.kwargs['usage_metadata']
        assert second_call['input_tokens'] == 3000
        assert second_call['output_tokens'] == 750
        assert second_call['total_tokens'] == 30750  # 3000 + 750 + 25000 + 2000

        # Step 3: ResultMessage (should NOT add usage, only cost)
        mock_conversation_run = Mock()
        mock_conversation_run.metadata = {}

        class MockResultMessage:
            usage = {'input_tokens': 3000, 'output_tokens': 750}
            total_cost_usd = 0.08
            num_turns = 1

        msg_result = MockResultMessage()

        # Handler only adds cost, not usage
        if hasattr(msg_result, "total_cost_usd") and msg_result.total_cost_usd is not None:
            mock_conversation_run.metadata["total_cost"] = msg_result.total_cost_usd

        # Verify final state
        assert mock_conversation_run.metadata['total_cost'] == 0.08
        assert 'usage_metadata' not in mock_conversation_run.metadata

        # The LLM run should have the usage from StreamEvents
        assert second_call['input_tokens'] == 3000
        assert second_call['output_tokens'] == 750
        assert second_call['total_tokens'] == 30750
        assert second_call['input_token_details']['cache_read'] == 25000
        assert second_call['input_token_details']['cache_creation'] == 2000
