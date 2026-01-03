"""Test that token accumulation from StreamEvents matches expected totals."""

from unittest.mock import Mock
import pytest
from langsmith.integrations.claude_agent_sdk._client import TurnLifecycle


class TestTokenAccumulation:
    """Test that buffering and accumulation produce correct token totals."""

    def test_single_turn_accumulation(self):
        """Test that a single turn accumulates tokens correctly."""
        tracker = TurnLifecycle()

        # Simulate message_start StreamEvent (arrives first, no run yet)
        message_start_usage = {
            'input_tokens': 1500,
            'output_tokens': 0,
            'total_tokens': 28000,
            'input_token_details': {
                'cache_read': 25000,
                'cache_creation': 1500,
            }
        }
        tracker.add_usage(message_start_usage)

        # Verify it was buffered (no run exists yet)
        assert tracker.pending_usage is not None
        assert tracker.current_run is None

        # Create run (simulating AssistantMessage arrival)
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()
        mock_run.end = Mock()
        mock_run.patch = Mock()

        from unittest.mock import patch
        with patch('langsmith.integrations.claude_agent_sdk._client.begin_llm_run_from_assistant_messages',
                  return_value=(None, mock_run)):
            tracker.start_llm_run(Mock(), "test", [])

        # Verify pending usage was applied
        assert tracker.pending_usage is None
        assert mock_run.set.call_count == 1

        # Now simulate message_delta StreamEvent (updates output tokens)
        message_delta_usage = {
            'output_tokens': 250,
            'total_tokens': 28250,  # 1500 + 250 + 25000 + 1500
        }
        tracker.add_usage(message_delta_usage)

        # Verify it was applied to current run (not buffered)
        assert mock_run.set.call_count == 2

        # Get final accumulated usage
        final_call = mock_run.set.call_args_list[1]
        final_usage = final_call.kwargs['usage_metadata']

        # Verify final totals
        assert final_usage['input_tokens'] == 1500
        assert final_usage['output_tokens'] == 250
        assert final_usage['total_tokens'] == 28250
        assert final_usage['input_token_details']['cache_read'] == 25000
        assert final_usage['input_token_details']['cache_creation'] == 1500

    def test_multi_turn_accumulation(self):
        """Test that multiple turns accumulate tokens correctly without cross-contamination."""
        tracker = TurnLifecycle()

        # ========== TURN 1 ==========
        # StreamEvent arrives first (no run)
        turn1_start = {
            'input_tokens': 1000,
            'output_tokens': 0,
            'total_tokens': 26000,
            'input_token_details': {'cache_read': 24000, 'cache_creation': 1000}
        }
        tracker.add_usage(turn1_start)
        assert tracker.pending_usage is not None

        # AssistantMessage arrives, creates run
        turn1_run = Mock()
        turn1_run.extra = {}
        turn1_run.set = Mock()
        turn1_run.end = Mock()
        turn1_run.patch = Mock()

        from unittest.mock import patch
        with patch('langsmith.integrations.claude_agent_sdk._client.begin_llm_run_from_assistant_messages',
                  return_value=(None, turn1_run)):
            tracker.start_llm_run(Mock(), "test", [])

        # message_delta updates output
        turn1_delta = {
            'output_tokens': 150,
            'total_tokens': 26150
        }
        tracker.add_usage(turn1_delta)

        # Get Turn 1 final usage
        turn1_final = turn1_run.set.call_args_list[-1].kwargs['usage_metadata']
        assert turn1_final['input_tokens'] == 1000
        assert turn1_final['output_tokens'] == 150
        assert turn1_final['total_tokens'] == 26150

        # ========== TURN 2 ==========
        # StreamEvent arrives (Turn 1's run still active!)
        turn2_start = {
            'input_tokens': 2000,
            'output_tokens': 0,
            'total_tokens': 27500,
            'input_token_details': {'cache_read': 25000, 'cache_creation': 500}
        }
        tracker.add_usage(turn2_start)

        # This should be buffered because Turn 1's run is still current
        # (In real flow, AssistantMessage for Turn 2 hasn't arrived yet)
        assert tracker.pending_usage is not None
        assert tracker.pending_usage['input_tokens'] == 2000

        # AssistantMessage for Turn 2 arrives, ends Turn 1, creates Turn 2
        turn2_run = Mock()
        turn2_run.extra = {}
        turn2_run.set = Mock()
        turn2_run.end = Mock()
        turn2_run.patch = Mock()

        with patch('langsmith.integrations.claude_agent_sdk._client.begin_llm_run_from_assistant_messages',
                  return_value=(None, turn2_run)):
            tracker.start_llm_run(Mock(), "test", [])

        # Verify Turn 1 was ended
        assert turn1_run.end.called
        assert turn1_run.patch.called

        # Verify Turn 2 pending usage was applied
        assert tracker.pending_usage is None
        turn2_initial = turn2_run.set.call_args_list[0].kwargs['usage_metadata']
        assert turn2_initial['input_tokens'] == 2000
        assert turn2_initial['total_tokens'] == 27500

        # message_delta for Turn 2
        turn2_delta = {
            'output_tokens': 300,
            'total_tokens': 27800
        }
        tracker.add_usage(turn2_delta)

        # Get Turn 2 final usage
        turn2_final = turn2_run.set.call_args_list[-1].kwargs['usage_metadata']
        assert turn2_final['input_tokens'] == 2000
        assert turn2_final['output_tokens'] == 300
        assert turn2_final['total_tokens'] == 27800

        # ========== VERIFY NO CROSS-CONTAMINATION ==========
        # Turn 1 should not have been affected by Turn 2's usage
        # (Turn 1 was already ended and patched before Turn 2 started)

        # Turn 1 final tokens
        assert turn1_final['input_tokens'] == 1000
        assert turn1_final['output_tokens'] == 150

        # Turn 2 final tokens (completely independent)
        assert turn2_final['input_tokens'] == 2000
        assert turn2_final['output_tokens'] == 300

        # Total across both turns
        total_input = turn1_final['input_tokens'] + turn2_final['input_tokens']
        total_output = turn1_final['output_tokens'] + turn2_final['output_tokens']

        assert total_input == 3000
        assert total_output == 450

    def test_message_delta_updates_total_correctly(self):
        """Test that message_delta recalculates total including cache tokens."""
        tracker = TurnLifecycle()

        # message_start with cache tokens
        start_usage = {
            'input_tokens': 500,
            'output_tokens': 0,
            'total_tokens': 10500,
            'input_token_details': {
                'cache_read': 9000,
                'cache_creation': 1000,
            }
        }
        tracker.add_usage(start_usage)

        # Create run
        mock_run = Mock()
        mock_run.extra = {}
        mock_run.set = Mock()
        mock_run.end = Mock()
        mock_run.patch = Mock()

        from unittest.mock import patch
        with patch('langsmith.integrations.claude_agent_sdk._client.begin_llm_run_from_assistant_messages',
                  return_value=(None, mock_run)):
            tracker.start_llm_run(Mock(), "test", [])

        # message_delta updates output tokens
        # Total should be: 500 input + 750 output + 9000 cache_read + 1000 cache_create = 11250
        delta_usage = {
            'output_tokens': 750,
            'total_tokens': 11250,
        }
        tracker.add_usage(delta_usage)

        # Verify final usage has correct total
        final_usage = mock_run.set.call_args_list[-1].kwargs['usage_metadata']
        assert final_usage['input_tokens'] == 500
        assert final_usage['output_tokens'] == 750
        assert final_usage['total_tokens'] == 11250
        assert final_usage['input_token_details']['cache_read'] == 9000
        assert final_usage['input_token_details']['cache_creation'] == 1000
