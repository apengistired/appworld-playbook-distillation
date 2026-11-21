import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
from io import StringIO

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_lesson
from harness_modules import ExecutionGuardrails, PlaybookEntry, ScopedPlaybookRouter, parse_playbook
from harness_modules import (
    build_audit_prompt, build_pairwise_judge_prompt, parse_audit_verdict,
    parse_pairwise_choice, parse_pairwise_verdict, summarize_attempt, audit_risk_reasons,
)


class ScopedRouterTests(unittest.TestCase):
    def test_parse_rendered_and_json_entries(self):
        text = (
            "## STRATEGIES_AND_HARD_RULES\n"
            "[spotify-page] helpful=7 harmful=2 :: Check Spotify pagination.\n"
            "  Use page_index when the response contains it.\n\n"
            "## VERIFICATION_CHECKLIST\n"
            "[gmail-label] helpful=3 harmful=0 :: Verify Gmail labels."
        )
        entries = parse_playbook(text)
        self.assertEqual(entries, [
            PlaybookEntry("spotify-page", "Check Spotify pagination.\n  Use page_index when the response contains it.", 7, 2),
            PlaybookEntry("gmail-label", "Verify Gmail labels.", 3),
        ])
        bullets = [entry.__dict__ for entry in entries]
        self.assertEqual(parse_playbook(json.dumps({"bullets": bullets})), entries)
        self.assertEqual(parse_playbook(json.dumps(bullets)), entries)
        self.assertEqual(parse_playbook("(empty)"), [])

    def test_app_and_error_match_with_topk(self):
        router = ScopedPlaybookRouter([
            PlaybookEntry("unrelated", "Resolve a Gmail KeyError while retrieving inbox labels.", 400),
            PlaybookEntry("page", "Read Spotify playlist results across pages before computing totals."),
            PlaybookEntry("error", "A KeyError in Spotify track results means inspect the schema."),
            PlaybookEntry("generic", "For Todoist, check time zones while filtering."),
        ], topk=1)
        context = (
            "List my Spotify playlist tracks",
            "print(apis.spotify.show_playlist_tracks())",
            "KeyError: track results are missing a field",
        )
        first = router.select_rules(*context)
        self.assertEqual([entry.id for entry in first], ["error"])
        self.assertIn("[error] A KeyError", router.format_hint(first))
        self.assertEqual(router.select_rules(*context), [])
        self.assertEqual(router.select_rules(*context, already_injected_ids={"error"}), [])
        self.assertEqual([entry.id for entry in router.select_rules(*context)], ["error"])

    def test_supervisor_and_example_apps_do_not_limit_rule_scope(self):
        router = ScopedPlaybookRouter([
            PlaybookEntry("supervisor", "Get credentials through apis.supervisor during authentication."),
            PlaybookEntry("parentheses", "Inspect login credentials for an API (e.g., Phone, Venmo)."),
            PlaybookEntry("standalone", "Inspect the login response, e.g., Phone or Venmo. Verify the token."),
            PlaybookEntry("scoped", "For Venmo authentication (e.g., Phone), inspect the login response.", 100),
        ], topk=4)
        rules = router.select_rules("Email my Amazon receipt")
        self.assertEqual({entry.id for entry in rules}, {"supervisor", "parentheses", "standalone"})

    def test_mid_task_hints_require_an_error_or_a_new_app(self):
        router = ScopedPlaybookRouter([
            PlaybookEntry("spotify", "For Spotify, use the playlist track IDs."),
            PlaybookEntry("venmo", "For Venmo, identify the recipient before a transfer."),
            PlaybookEntry("lookup-error", "If a KeyError occurs in Spotify, inspect the response schema."),
            PlaybookEntry("value-error", "For a Spotify ValueError, check the response schema."),
        ])
        task = "Find my Spotify tracks"
        self.assertTrue(router.select_rules(task))
        self.assertEqual(router.select_rules(task, "print('progress')", "next"), [])
        # A mention in the task text does not substitute for the first action in that app.
        spotify = "apis.spotify.show_tracks()"
        self.assertTrue(router.select_rules(task, spotify, "Tracks returned"))
        self.assertEqual(router.select_rules(task, spotify, "More tracks"), [])
        new = router.select_rules(task, "apis.venmo.show_transactions()", "Transactions returned")
        self.assertEqual([entry.id for entry in new], ["venmo"])
        error = router.select_rules(task, spotify, "KeyError: unknown response schema")
        self.assertEqual([entry.id for entry in error], ["lookup-error"])

    def test_verification_observation_suppresses_error_and_new_app_hints(self):
        router = ScopedPlaybookRouter([
            PlaybookEntry("error", "Fix a KeyError in Venmo by inspecting the schema."),
        ])
        observation = "[Harness Verification Required]\nKeyError: check Venmo result"
        code = "apis.venmo.show_transactions()"
        self.assertEqual(router.select_rules("Verify a charge", code, observation), [])
        self.assertEqual([entry.id for entry in router.select_rules("Verify a charge", code, "Returned")], ["error"])

    def test_no_matching_topic_no_hint_and_duplicate_ids_rejected(self):
        rule = PlaybookEntry("same", "Verify the recipient when paying with Venmo.")
        router = ScopedPlaybookRouter([rule])
        self.assertEqual(router.select_rules("Inspect my Amazon orders"), [])
        self.assertEqual(router.format_hint([]), "")
        with self.assertRaisesRegex(ValueError, "unique"):
            ScopedPlaybookRouter([rule, rule])


class GuardrailTests(unittest.TestCase):
    write = "apis.gmail.update_label(label_id=2, name='Receipts')"
    read = "print(apis.gmail.show_label(label_id=2))"
    submit = "apis.supervisor.complete_task(answer=None)"

    def test_unverified_write_is_blocked_once(self):
        guard = ExecutionGuardrails("full")
        guard.after_execute(self.write, "Label updated")
        notice = guard.before_execute(self.submit)
        self.assertIn("[Harness Verification Required]", notice)
        self.assertIn("persisted result", notice)
        self.assertIsNone(guard.before_execute(self.submit))
        self.assertEqual(guard.stats(), {
            "auto_doc_injections": 0,
            "pagination_nudges": 0,
            "readback_intercepts": 1,
            "submit_gate_intercepts": 1,
            "mutating_calls_seen": 1,
            "read_calls_after_last_write": 0,
        })

    def test_readback_after_last_write_with_null_answer_is_allowed(self):
        guard = ExecutionGuardrails("full")
        guard.after_execute(self.read + "\n" + self.write, "Label updated")
        self.assertEqual(guard.stats()["read_calls_after_last_write"], 0)
        guard.after_execute(self.read, "{'name': 'Receipts'}")
        self.assertEqual(guard.stats()["read_calls_after_last_write"], 1)
        self.assertIsNone(guard.before_execute(self.submit))
        self.assertEqual(guard.stats()["submit_gate_intercepts"], 0)

    def test_non_null_answer_and_pagination_update_separate_counters(self):
        for submit in ("apis.supervisor.complete_task(answer=0)", "supervisor.complete_task('done')"):
            with self.subTest(submit=submit):
                guard = ExecutionGuardrails("full")
                nudge = guard.after_execute(
                    "print(apis.spotify.show_tracks())", json.dumps({"has_more": True}),
                )
                self.assertIn("[Harness Guardrail]", nudge)
                self.assertIn("answer=None", guard.before_execute(submit))
                self.assertIsNone(guard.before_execute(submit))
                self.assertEqual(guard.stats()["pagination_nudges"], 1)
                self.assertEqual(guard.stats()["submit_gate_intercepts"], 1)
                self.assertEqual(guard.stats()["readback_intercepts"], 0)

    def test_pagination_has_per_api_dedup_and_two_nudge_cap(self):
        guard = ExecutionGuardrails("basic")
        for observation in (
            '{"items":[1,2,3,4,5],"page_index":0,"next_page":null,"has_more":false}',
            "{'next_page': None, 'next_cursor': '', 'has_more': False, 'truncated': False}",
            '[{"song_title":"truncated"}]',
        ):
            with self.subTest(observation=observation):
                self.assertIsNone(guard.after_execute(self.read, observation))
        self.assertIn("[Harness Guardrail]", guard.after_execute(self.read, "has_more: True"))
        self.assertIsNone(guard.after_execute(self.read, '{"next_page":2}'))
        self.assertIn("[Harness Guardrail]", guard.after_execute("apis.gmail.search_messages()", "Found 10 messages"))
        self.assertIsNone(guard.after_execute("apis.spotify.show_tracks()", "Output truncated [...]"))
        self.assertEqual(guard.stats()["pagination_nudges"], 2)
        fresh = ExecutionGuardrails("basic")
        self.assertIn("[Harness Guardrail]", fresh.after_execute(self.read, "Output truncated [...]"))

    def test_non_loop_default_page_detection(self):
        for output in ("Retrieved 5 songs", "Found 10 matches", "[1,2,3,4,5]", json.dumps(list(range(20)))):
            with self.subTest(output=output):
                guard = ExecutionGuardrails("basic")
                code = "apis.spotify.show_song_library()"
                self.assertIn("[Harness Guardrail]", guard.after_execute(code, output))
                self.assertIsNone(guard.after_execute(code, output))

    def test_loop_and_explicit_paging_do_not_trigger(self):
        codes = (
            "for i in range(3):\n    apis.spotify.show_song_library()",
            "while more:\n    apis.spotify.show_song_library()",
            "[apis.spotify.show_song_library() for i in range(3)]",
            "apis.spotify.show_song_library(page_limit=20)",
            "apis.spotify.show_song_library(page_index=0)",
            "apis.spotify.show_song_library(offset=0)",
            "apis.spotify.show_song_library(cursor=None)",
        )
        for code in codes:
            with self.subTest(code=code):
                guard = ExecutionGuardrails("basic")
                self.assertIsNone(guard.after_execute(code, "Retrieved 5 songs [...]"))
        guard = ExecutionGuardrails("basic")
        code = "for x in range(2):\n    apis.spotify.show_tracks()\napis.gmail.search_messages()"
        self.assertIn("[Harness Guardrail]", guard.after_execute(code, "Found 5 matches"))
        self.assertEqual(guard._nudged_apis, {"apis.gmail.search_messages"})

    def test_extended_write_verbs_are_tracked(self):
        for verb in (
            "like_song", "unlike_song", "follow_artist", "unfollow_artist", "download_song", "upload_file",
            "move_file", "rename_file", "remind_user", "reply_message", "mark_message", "star_message",
            "archive_message", "enroll_course", "cancel_order",
        ):
            with self.subTest(verb=verb):
                guard = ExecutionGuardrails("full")
                guard.after_execute(f"apis.spotify.{verb}()", "OK")
                self.assertEqual(guard.stats()["mutating_calls_seen"], 1)
                self.assertIn("persisted result", guard.before_execute(self.submit))

    def test_fail_and_zero_mutation_get_one_repair_notice(self):
        for task, code in (
            ("Like my favorite songs", self.submit),
            ("How many songs?", "apis.supervisor.complete_task(status='fail')"),
        ):
            with self.subTest(task=task):
                guard = ExecutionGuardrails("full", task_text=task)
                notice = guard.before_execute(code)
                self.assertIn("email vs phone_number", notice)
                self.assertIn("show_artist_songs vs show_song_library", notice)
                self.assertIsNone(guard.before_execute(code))

    def test_query_answer_is_allowed_only_without_writes(self):
        guard = ExecutionGuardrails("full", task_text="How many songs are in the playlist?")
        non_null = "apis.supervisor.complete_task(answer=1)"
        guard.after_execute("apis.spotify.show_tracks()", "Tracks")
        self.assertIsNone(guard.before_execute(non_null))
        self.assertEqual(guard.stats()["submit_gate_intercepts"], 0)
        guard.after_execute("apis.spotify.add_song()", "Added")
        self.assertIn("persisted result", guard.before_execute(non_null))

    def test_action_confirmation_is_intercepted_even_with_question_in_message(self):
        guard = ExecutionGuardrails("full", task_text='Send Alex the message "Where are you?"')
        notice = guard.before_execute("apis.supervisor.complete_task(answer='Done')")
        self.assertIn("MUST submit answer=None", notice)
        self.assertIn("confirmation message", notice)
        self.assertIn('"26"', notice)
        self.assertIn("explicitly asks", notice)
        self.assertEqual(guard.stats()["submit_gate_intercepts"], 1)

    def test_basic_tracks_without_intercept_and_none_is_inert(self):
        for mode, writes, nudges in (("basic", 1, 1), ("none", 0, 0)):
            with self.subTest(mode=mode):
                guard = ExecutionGuardrails(mode)
                guard.after_execute(self.read, "{'page_index': 0, 'next_page': 1}")
                guard.after_execute(self.write, "Label updated")
                self.assertIsNone(guard.before_execute(self.submit))
                self.assertEqual(guard.stats()["mutating_calls_seen"], writes)
                self.assertEqual(guard.stats()["pagination_nudges"], nudges)

    def test_failed_read_does_not_verify_and_paging_loop_is_not_nudged(self):
        guard = ExecutionGuardrails("full")
        guard.after_execute(self.write, "Label updated")
        guard.after_execute(self.read, "Execution failed. Traceback:\nKeyError: page_index")
        self.assertEqual(guard.stats()["read_calls_after_last_write"], 0)
        self.assertEqual(guard.stats()["pagination_nudges"], 0)
        self.assertIsNotNone(guard.before_execute(self.submit))
        loop = "for page in range(2):\n    print(apis.gmail.list_labels(page_index=page))"
        self.assertIsNone(guard.after_execute(loop, "{'next_page': 2}"))

    def test_gate_can_hold_a_combined_write_and_submit_without_counting_held_write(self):
        guard = ExecutionGuardrails("full")
        notice = guard.before_execute(self.write + "\n" + self.submit)
        self.assertIn("This code cell was held", notice)
        self.assertEqual(guard.stats()["mutating_calls_seen"], 0)
        fresh = ExecutionGuardrails("full")
        self.assertIsNone(fresh.before_execute("# apis.supervisor.complete_task(42)\nprint('complete_task(42)')"))
        self.assertEqual(fresh.stats()["submit_gate_intercepts"], 0)


class RSIGuardTests(unittest.TestCase):
    def guard(self):
        return ExecutionGuardrails("full", auto_doc=False, rsi_guards="v5")

    def test_defaults_preserve_six_statistics_and_new_modes_are_validated(self):
        original = ExecutionGuardrails("full")
        guarded = self.guard()
        self.assertEqual(set(original.stats()), set(guarded.stats()))
        self.assertEqual(len(guarded.stats()), 6)
        self.assertTrue(all(count == 0 for count in guarded.rsi_stats().values()))
        with self.assertRaises(ValueError):
            ExecutionGuardrails(rsi_guards="unknown")
        with self.assertRaises(ValueError):
            ExecutionGuardrails(auto_doc="none")
        docs = Mock()
        self.assertIsNone(guarded.auto_doc_on_error("apis.phone.get_contacts()", "Execution failed. Traceback:", docs))
        docs.show_api_doc.assert_not_called()
        docs.show_api_descriptions.assert_not_called()

    def test_pagination_loop_mismatched_default_and_per_api_dedup(self):
        code = "while True:\n    page = apis.spotify.show_song_library(page_index=i)\n    if not page: break\n    if len(page) < 20: break"
        guard = self.guard()
        self.assertIn("page_limit=5", guard.after_execute(code, "Retrieved 5 songs"))
        self.assertIsNone(guard.after_execute(code, "Retrieved 5 songs"))
        self.assertEqual(guard.rsi_stats()["pagination_limit_guard"], 1)
        self.assertIn("page_limit=5", self.guard().after_execute("apis.spotify.show_song_library()", "[1,2,3,4,5]"))

    def test_pagination_does_not_nudge_explicit_limits_exhaustion_or_unrelated_lengths(self):
        for code in (
            "while True:\n    page = apis.spotify.show_song_library(page_index=i, page_limit=20)\n    if len(page) < 20: break",
            "while True:\n    page = apis.spotify.show_song_library(page_index=i)\n    if not page: break",
            "while True:\n    page = apis.spotify.show_song_library(page_index=i)\n    if len(page) < 5: break",
            "while True:\n    page = apis.spotify.show_song_library(page_index=i)\n    if len(other_records) < 20: break",
            "apis.spotify.show_song_library(page_limit=20)",
            "apis.spotify.show_genres()",
        ):
            with self.subTest(code=code):
                guard = self.guard()
                guard.after_execute(code, "[1,2,3,4,5]")
                self.assertEqual(guard.rsi_stats()["pagination_limit_guard"], 0)

    def test_playlist_rating_requires_song_details_and_tracks_previous_steps(self):
        guard = self.guard()
        guard.after_execute("playlist = apis.spotify.show_playlist(playlist_id=7)", "{}")
        hint = guard.after_execute("rated = [s for s in playlist['songs'] if s.get('rating', 0) > 0]", "[]")
        self.assertIn("show_song(song_id)", hint)
        self.assertIsNone(guard.after_execute("rating = song['rating']", "0"))
        safe = self.guard()
        safe.after_execute("playlist = apis.spotify.show_playlist(playlist_id=7)\ninfo = apis.spotify.show_song(song_id=8)", "{}")
        self.assertIsNone(safe.after_execute("rating = info.get('rating', 0)", "0"))

    def test_insufficient_balance_recommends_direct_card_and_deduplicates(self):
        guard = self.guard()
        code = "apis.venmo.create_transaction(amount=12)"
        error = "Execution failed. Traceback:\nException: Response status code is 422: Your Venmo balance does not have enough money."
        hint = guard.after_execute(code, error)
        self.assertIn("show_payment_cards()", hint)
        self.assertIn("payment_card_id", hint)
        self.assertIn("avoid add_to_venmo_balance", hint)
        self.assertIsNone(guard.after_execute(code, error))
        self.assertEqual(guard.rsi_stats()["venmo_card_and_batch_idempotency"], 1)
        for observation in ("422: payment card does not exist", "401: insufficient balance", "Sent money"):
            self.assertIsNone(self.guard().after_execute(code, observation))

    def test_already_existing_batch_state_recommends_per_item_recovery(self):
        code = "for song in songs:\n    apis.spotify.download_song(song_id=song['song_id'])"
        error = "Execution failed. Traceback:\nException: 422 already downloaded"
        guard = self.guard()
        self.assertIn("per-item try/except", guard.after_execute(code, error))
        self.assertIsNone(guard.after_execute(code, error))
        safe = "for song in songs:\n    try:\n        apis.spotify.download_song(song_id=song['song_id'])\n    except Exception:\n        continue"
        self.assertIsNone(self.guard().after_execute(safe, error))
        self.assertIsNone(self.guard().after_execute("apis.spotify.download_song(song_id=1)", error))
        self.assertIsNone(self.guard().after_execute(code, "422 invalid song"))

    def test_fuzzy_user_existence_needs_exact_returned_identity(self):
        unsafe = "users = apis.venmo.search_users(query=target)\nif len(users) > 0:\n    found = True"
        guard = self.guard()
        self.assertIn("exact match", guard.after_execute(unsafe, "Found"))
        self.assertIsNone(guard.after_execute(unsafe, "Found"))
        for safe in (
            "users = apis.venmo.search_users(query=target)\nif users:\n    matches = [u for u in users if u['email'] == target]",
            "users = apis.venmo.search_users(query=target)\nfor user in users:\n    if user.get('name', '').lower() == target.lower():\n        found = True",
            "users = apis.venmo.search_users(query=target)\nprint(users)",
        ):
            self.assertIsNone(self.guard().after_execute(safe, "Found"))
        unrelated = unsafe + "\nif profile['email'] == target: print('profile')"
        self.assertIn("exact match", self.guard().after_execute(unrelated, "Found"))

    def test_message_query_warns_once_but_complete_conversation_is_not_flagged(self):
        guard = self.guard()
        query = "apis.phone.search_text_messages(phone_number=phone, query='groceries')"
        self.assertIn("without query", guard.after_execute(query, "[]"))
        self.assertIsNone(guard.after_execute(query, "[]"))
        for code in ("apis.phone.search_text_messages(phone_number=phone)",
                     "apis.phone.search_text_messages(phone_number=phone, query='')"):
            self.assertIsNone(self.guard().after_execute(code, "[]"))

    def test_batch_date_prefix_uses_file_metadata_and_is_deduplicated(self):
        wrong = "current_date = datetime.now()\nprefix = current_date.strftime('%Y_%m_%d')\nfor file in files:\n    destination = f\"{prefix}_{file['name']}\"\n    apis.file_system.move_file(source_file_path=file['path'], destination_file_path=destination)"
        guard = self.guard()
        hint = guard.after_execute(wrong, "Moved files")
        self.assertIn("created_at rather than", hint)
        self.assertIn("double prefixes", hint)
        self.assertIsNone(guard.after_execute(wrong, "Moved files"))
        safe = "current_date = datetime.now()\nfor file in files:\n    created = datetime.fromisoformat(file['created_at'])\n    prefix = created.strftime('%Y_%m_%d')\n    if file['name'].startswith(prefix): continue\n    destination = f\"{prefix}_{file['name']}\"\n    apis.file_system.move_file(source_file_path=file['path'], destination_file_path=destination)"
        self.assertIsNone(self.guard().after_execute(safe, "Moved files"))

    def test_rename_failure_carries_risk_to_next_step_and_relist_is_deduplicated(self):
        guard = self.guard()
        move = "apis.file_system.move_file(source_file_path=old, destination_file_path=new)"
        self.assertIn("re-list", guard.after_execute(move, "Execution failed. Traceback:\n422 file unavailable"))
        self.assertIsNone(guard.after_execute("apis.file_system.show_directory(directory_path=folder)", "{}"))
        clean = self.guard()
        self.assertIsNone(clean.after_execute(move, "Moved"))
        self.assertIn("double prefixes", clean.after_execute(move, "Moved"))

    def test_none_mode_remains_inert_unless_rsi_is_explicitly_requested(self):
        code = "while True:\n    page = apis.spotify.show_song_library(page_index=i)\n    if len(page) < 20: break"
        self.assertIsNone(ExecutionGuardrails().after_execute(code, "Found 5 songs"))
        guard = ExecutionGuardrails("none", rsi_guards="v5")
        self.assertIn("[RSI Guard]", guard.after_execute(code, "Found 5 songs"))
        self.assertTrue(all(value == 0 for value in guard.stats().values()))


class AutoDocTests(unittest.TestCase):
    def test_parameter_and_422_doc_constraints_and_per_api_dedup(self):
        docs = Mock()
        docs.show_api_doc.return_value = {"api_name": "move_file", "parameters": [
            {"name": "file_path", "constraints": ["must exist"]},
        ]}
        guard = ExecutionGuardrails("full")
        for api, error in (("move_file", "Unexpected parameter 'source'"), ("rename_file", "422 invalid path")):
            code = f"apis.file_system.{api}(source='x')"
            obs = "Execution failed. Traceback:\nTypeError: " + error
            notice = guard.auto_doc_on_error(code, obs, docs)
            self.assertIn("[Harness API Doc]", notice)
            self.assertIn("must exist", notice)
            docs.show_api_doc.assert_called_with(app_name="file_system", api_name=api)
            self.assertIsNone(guard.auto_doc_on_error(code, obs, docs))
        self.assertEqual(guard.stats()["auto_doc_injections"], 2)
        self.assertEqual(docs.show_api_doc.call_count, 2)

    def test_unknown_api_uses_live_descriptions_and_traceback_path_wins(self):
        docs = Mock()
        docs.show_api_descriptions.return_value = [{"name": "search_contacts", "description": "Find contacts"}]
        guard = ExecutionGuardrails("full")
        obs = "Execution failed. Traceback:\napis.phone.get_contacts()\nAttributeError: has no attribute get_contacts"
        notice = guard.auto_doc_on_error("apis.gmail.search_messages()", obs, docs)
        self.assertIn("search_contacts", notice)
        docs.show_api_descriptions.assert_called_once_with(app_name="phone")
        docs.show_api_doc.assert_not_called()
        self.assertIsNone(guard.auto_doc_on_error("apis.phone.get_contacts()", obs, docs))

    def test_lookup_failure_missing_path_and_none_mode_preserve_observation(self):
        docs = Mock()
        docs.show_api_doc.side_effect = ValueError("Unavailable")
        guard = ExecutionGuardrails("full")
        for code, obs in (("print(1)", "Execution failed. Traceback:\nNameError: x"),
                          ("apis.phone.get_contacts()", "OK"),
                          ("apis.phone.get_contacts()", "Execution failed. Traceback:\nTypeError: bad")):
            self.assertIsNone(guard.auto_doc_on_error(code, obs, docs))
        self.assertEqual(guard.stats()["auto_doc_injections"], 0)
        docs.reset_mock()
        self.assertIsNone(ExecutionGuardrails().auto_doc_on_error(
            "apis.phone.get_contacts()", "Execution failed. Traceback:\n422 bad", docs,
        ))
        docs.show_api_doc.assert_not_called()


class AuditTests(unittest.TestCase):
    def test_entire_readback_and_long_code_survive_both_prompts(self):
        rows = [{"song_id": i, "artist": "Existing artist"} for i in range(60)]
        rows.append({"song_id": 999, "artist": "Requested artist"})
        observation = "Current queue: " + repr(rows)
        code = "# explanatory comment\n" * 100 + "apis.spotify.add_to_queue(song_id=999)"
        summary = summarize_attempt([
            {"input": code, "output": "Added song with ID 999 to the queue."},
            {"input": "print(apis.spotify.show_song_queue())", "output": observation},
        ], {}, None)
        self.assertEqual(summary["excerpts"][0]["code"], code)
        self.assertEqual(summary["excerpts"][1]["observation"], observation)
        self.assertFalse(summary["summary_truncated"])
        self.assertTrue(summary["excerpts"][1]["observation_complete"])
        for prompt in (build_audit_prompt("Add requested songs", summary),
                       build_pairwise_judge_prompt("Add requested songs", summary, summary)):
            self.assertIn("'song_id': 999", prompt)
            self.assertIn("Existing artist", prompt)
            self.assertIn("Requested artist", prompt)
        self.assertIn("extra unrelated changes", build_pairwise_judge_prompt("Add songs", summary, summary))

    def test_environment_truncation_is_explicit_and_held_writes_are_not_counted(self):
        summary = summarize_attempt([
            {"input": "apis.spotify.add_to_queue(song_id=5)",
             "output": "[Harness Verification Required] Cell was held"},
            {"input": "print(apis.spotify.show_song_queue())", "output": "Results truncated [...]"},
        ], {}, None)
        self.assertEqual(summary["visible_write_calls"], 0)
        self.assertFalse(summary["excerpts"][1]["observation_complete"])

    def test_pairwise_requires_real_step_quotes_from_both_attempts(self):
        first = summarize_attempt([{"input": "print('old')", "output": "Required item missing from readback."}], {}, None)
        second = summarize_attempt([{"input": "print('new')", "output": "Required item present in full readback."}], {}, None)
        verdict = {"choice": 2, "confidence": "high", "reason": "Second attempt fixes missing item.",
                   "preferred_evidence": {"step": 1, "field": "observation", "quote": "Required item present"},
                   "rejected_issue": {"step": 1, "field": "observation", "quote": "Required item missing"}}
        self.assertTrue(parse_pairwise_verdict(json.dumps(verdict), first, second)["verified"])
        for update in ({"choice": True}, {"confidence": "low"}, {"reason": ""},
                       {"preferred_evidence": {"step": 20, "field": "observation", "quote": "Required item present"}},
                       {"rejected_issue": {"step": 1, "field": "observation", "quote": "Invented evidence from nowhere"}}):
            parsed = parse_pairwise_verdict(json.dumps({**verdict, **update}), first, second)
            self.assertFalse(parsed["verified"])
            self.assertEqual(parsed["choice"], 1)
        self.assertFalse(parse_pairwise_verdict('{"choice":2}', first, second)["verified"])

    def test_reasoning_and_markdown_final_json(self):
        for prefix, suffix in (("<think>Ignore {\"choice\":1}</think>\n", ""), ("```json\n", "\n```"),
                               ("<think>Reasoning</think>\n```json\n", "\n```")):
            self.assertEqual(parse_pairwise_choice(prefix + '{"choice":2}' + suffix), 2)
            self.assertEqual(parse_audit_verdict(prefix + '{"verdict":"RETRY","critique":"Fix source."}' + suffix)
                             ["verdict"], "RETRY")
        self.assertEqual(parse_pairwise_choice('<think>{"choice":2}'), 1)
        self.assertEqual(parse_pairwise_choice('{"choice":2} trailing junk'), 1)

    def test_all_middle_code_and_all_complete_errors_are_present(self):
        steps = [{"input": f"print('middle_step_{i}')", "output": "OK"} for i in range(15)]
        error = "Execution failed. Traceback:\nUnexpected parameter 'source' " + "detail " * 400
        steps[7] = {"input": "apis.file_system.move_file(source='x')", "output": error}
        steps[8] = {"input": "apis.file_system.show_files()", "output": "Execution failed. Traceback:\n422 invalid"}
        summary = summarize_attempt(steps, {}, None)
        self.assertEqual(len(summary["excerpts"]), 15)
        self.assertIn("middle_step_6", summary["excerpts"][6]["code"])
        self.assertEqual(summary["excerpts"][7]["observation"], error)
        self.assertIn("Step 8: apis.file_system.move_file ->", summary["discovered_errors"][0])
        self.assertTrue(summary["err_after_write"])
        self.assertEqual(summary["execution_error_steps"], [8, 9])

    def test_risk_only_gates_on_each_objective_signal(self):
        clean = summarize_attempt([{"input": "print(3)", "output": "3"}], {}, None)
        self.assertEqual(audit_risk_reasons("How many songs?", clean), [])
        cases = [
            ({"fail_seen": True}, "explicit_fail"), ({"error": "Oops"}, "runner_error"),
            ({"guardrails": {"readback_intercepts": 1}}, "readback_intercept"),
            ({"err_after_write": True}, "error_after_write"),
            ({"execution_error_steps": [1, 2]}, "multiple_execution_errors"),
            ({"num_steps": 40}, "long_trajectory"),
        ]
        for changes, reason in cases:
            self.assertIn(reason, audit_risk_reasons("How many songs?", {**clean, **changes}))
        self.assertIn("action_without_mutation", audit_risk_reasons("Move my files", clean))
        self.assertNotIn("action_without_mutation", audit_risk_reasons(
            "Move my files", {**clean, "guardrails": {"mutating_calls_seen": 1}},
        ))

    def test_strict_parsers_and_conservative_fallbacks(self):
        self.assertEqual(parse_audit_verdict('{"verdict":"RETRY","critique":"Check the next page."}'),
                         {"verdict": "RETRY", "critique": "Check the next page."})
        self.assertEqual(parse_audit_verdict('{"verdict":"PASS","critique":"Complete."}')["verdict"], "PASS")
        for raw in ("broken", "[]", "null", '{"verdict":"RETRY"}', '{"verdict":"RETRY","critique":false}',
                    '{"verdict":"RETRY","critique":""}', '{"verdict":"OTHER","critique":"x"}'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_audit_verdict(raw), {"verdict": "PASS", "critique": ""})
        self.assertEqual(parse_pairwise_choice('{"choice":2}'), 2)
        for raw in ("broken", "[]", "null", '{"choice":true}', '{"choice":"2"}', '{"choice":2.0}', '{"choice":3}'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_pairwise_choice(raw), 1)

    def test_large_observations_are_not_silently_truncated(self):
        steps = [{"input": "print('x')", "output": "x" * 10000} for _ in range(60)]
        steps[30] = {"input": "apis.venmo.transfer_money()", "output": "Execution failed. Traceback:\n422 insufficient balance"}
        steps[-1] = {"input": "apis.supervisor.complete_task(status='fail')", "output": "Completed"}
        summary = summarize_attempt(steps, ExecutionGuardrails().stats(), None)
        self.assertEqual(summary["final_status"], "fail")
        self.assertEqual(summary["execution_error_steps"], [31])
        prompt = build_audit_prompt("Transfer money", summary)
        self.assertIn("422 insufficient balance", prompt)
        self.assertEqual(summary["excerpts"][20]["observation"], "x" * 10000)
        self.assertEqual(len(summary["excerpts"]), 60)
        self.assertIn("attempt_2", build_pairwise_judge_prompt("Transfer money", summary, summary))


class SelectiveRetryTests(unittest.TestCase):
    def test_risk_only_skips_clean_query_without_a_critic_call(self):
        row, events = self.run_job([], retry_trigger="risk_only", instruction="How many songs?")
        self.assertEqual([event[0] for event in events], ["run", "evaluate"])
        self.assertEqual(row["retry"], {"audited": False, "triggered": False, "selected": 1, "critique": ""})

    def test_risk_only_audits_zero_mutation_action(self):
        row, events = self.run_job(['{"verdict":"PASS","critique":"No matching files."}'],
                                   retry_trigger="risk_only")
        self.assertEqual([event[0] for event in events], ["run", "audit", "evaluate"])
        self.assertTrue(row["retry"]["audited"])

    def test_retry_temperature_seed_do_not_change_first_attempt_config(self):
        module = ModuleType("appworld")
        module.AppWorld = Mock()
        module.AppWorld.parse_environment_io_log.return_value = []
        cfg = {"options": {"retry_mode": "selective", "retry_temperature": 0.65},
               "agent": {"model_config": {"temperature": 0, "seed": 100}}, "header": ""}
        agent = Mock(guardrail_state=None, step_number=0)
        with (patch.dict(sys.modules, {"appworld": module}), patch.dict(run_lesson._CFG, cfg, clear=True),
              patch.object(run_lesson, "make_agent", return_value=agent) as make):
            run_lesson._run_attempt("task", "p2", "", retry=True)
        self.assertEqual(make.call_args.args[0]["model_config"], {"temperature": 0.65, "seed": 101})
        self.assertEqual(cfg["agent"]["model_config"], {"temperature": 0, "seed": 100})

    def test_critic_uses_thinking_base_without_mutating_actor_options(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"choice":2}'))],
        )
        module = ModuleType("openai")
        module.OpenAI = Mock(return_value=client)
        cfg = {"options": {}, "agent": {"model_config": {
            "name": "ace20", "base_url": "http://local/v1", "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }}}
        with (patch.dict(sys.modules, {"openai": module}), patch.dict(run_lesson._CFG, cfg, clear=True),
              patch.object(run_lesson, "_critic_prompt_size", return_value=(100, 65536))):
            self.assertEqual(run_lesson._critic("prompt"), '{"choice":2}')
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(module.OpenAI.call_args.kwargs["timeout"], 180)
        self.assertEqual(request["model"], "qwen3.5-9b")
        self.assertEqual(request["max_tokens"], 8192)
        self.assertTrue(request["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(cfg["agent"]["model_config"]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def attempt(self, exp, status="success", error=None):
        return {
            "exp": exp, "steps": [
                {"input": "print('state readback')", "output": "Remaining required changes: " + ("0" if exp.endswith("_retry") else "1")},
                {"input": f"apis.supervisor.complete_task(status={status!r})", "output": "Completed"},
            ],
            "error": error, "guardrails": ExecutionGuardrails().stats(), "num_agent_steps": 1,
        }

    @staticmethod
    def judge_reply(choice, prefer_retry=True):
        return json.dumps({
            "choice": choice, "confidence": "high", "reason": "Compare remaining required changes in state readback.",
            "preferred_evidence": {"step": 1, "field": "observation",
                                   "quote": "Remaining required changes: " + ("0" if prefer_retry else "1")},
            "rejected_issue": {"step": 1, "field": "observation",
                               "quote": "Remaining required changes: " + ("1" if prefer_retry else "0")},
        })

    def run_job(self, verdicts, first_status="success", error=None, retry_mode="selective",
                retry_trigger="all_audited", instruction="Move the file"):
        events = []
        exp = "spine/gate1/a_r1"
        task_module = ModuleType("appworld.task")
        task_object = Mock(instruction=instruction)
        task_module.Task = Mock()
        task_module.Task.load.return_value = task_object
        evaluator = ModuleType("appworld.evaluator")

        def run(task_id, name, lesson, extra="", *, retry=False):
            self.assertEqual(retry, name.endswith("_retry"))
            events.append(("run", name, extra))
            return self.attempt(name, first_status if name == exp else "success", error if name == exp else None)

        answers = iter(verdicts)
        def critic(prompt):
            self.assertNotIn("EVALUATION_SECRET", prompt)
            events.append(("judge" if "attempt_2" in prompt else "audit",))
            return next(answers)

        def evaluate(task_id, name):
            events.append(("evaluate", name))
            return SimpleNamespace(to_dict=lambda: {
                "success": True, "num_tests": 1, "failures": [], "difficulty": "EVALUATION_SECRET",
            })
        evaluator.evaluate_task = Mock(side_effect=evaluate)
        config = {"options": {"retry_mode": retry_mode, "retry_trigger": retry_trigger}, "agent": {}, "header": ""}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(run_lesson, "ROOT", Path(directory)),
            patch.dict(run_lesson._CFG, config, clear=True),
            patch.dict(sys.modules, {"appworld.task": task_module, "appworld.evaluator": evaluator}),
            patch.object(run_lesson, "_run_attempt", side_effect=run),
            patch.object(run_lesson, "_critic", side_effect=critic) as mocked_critic,
        ):
            row = run_lesson._job(("a", "r1", "task", ""))
            evaluator.evaluate_task.assert_called_once()
            if retry_mode == "selective":
                task_module.Task.load.assert_called_once_with(task_id="task", load_ground_truth=False)
                task_object.close.assert_called_once()
            else:
                mocked_critic.assert_not_called()
            return row, events

    def test_retry_selects_second_before_only_evaluation(self):
        row, events = self.run_job(['{"verdict":"RETRY","critique":"Use a complete search."}',
                                   self.judge_reply(2), self.judge_reply(1)])
        self.assertEqual([event[0] for event in events], ["run", "audit", "run", "judge", "judge", "evaluate"])
        self.assertIn("environment has been reset", events[2][2])
        self.assertIn("Use a complete search.", events[2][2])
        self.assertEqual(events[-1], ("evaluate", "spine/gate1/a_r1_retry"))
        self.assertEqual(row["retry"]["selected"], 2)
        self.assertEqual(row["retry"]["critique"], "Use a complete search.")
        self.assertEqual(row["retry"]["selection"]["method"], "evidence_order_consensus")
        self.assertEqual(row["selected_exp"], "spine/gate1/a_r1_retry")

    def test_order_disagreement_and_missing_evidence_keep_first(self):
        for answers in ([self.judge_reply(2), self.judge_reply(2, prefer_retry=False)],
                        [self.judge_reply(2), "malformed"], ['{"choice":2}']):
            row, events = self.run_job(['{"verdict":"RETRY","critique":"Check remaining changes."}', *answers])
            self.assertEqual(row["retry"]["selected"], 1)
            self.assertEqual(events[-1], ("evaluate", "spine/gate1/a_r1"))

    def test_critic_context_limit_preserves_complete_prompt_and_logs_skip(self):
        self.check_critic_trace((64000, 65536), "stop", "context_exceeded", "")

    def test_critic_truncated_reply_is_saved_and_not_used_for_selection(self):
        self.check_critic_trace((100, 65536), "length", "output_truncated", "")

    def check_critic_trace(self, tokens, finish_reason, expected_status, expected_return):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"choice":2}'), finish_reason=finish_reason,
        )])
        module = ModuleType("openai")
        module.OpenAI = Mock(return_value=client)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "critic.jsonl"
            cfg = {"options": {}, "agent": {"model_config": {"base_url": "http://local/v1"}},
                   "critic_trace": str(path), "critic_phase": "judge_reverse"}
            prompt = "long runtime evidence " * 1000 + "CORRECT_NEW_ITEM_AT_END"
            with (patch.dict(sys.modules, {"openai": module}), patch.dict(run_lesson._CFG, cfg, clear=True),
                  patch.object(run_lesson, "_critic_prompt_size", return_value=tokens)):
                self.assertEqual(run_lesson._critic(prompt), expected_return)
            trace = json.loads(path.read_text())
            self.assertEqual(trace["prompt"], prompt)
            self.assertEqual(trace["status"], expected_status)
            self.assertEqual(trace["phase"], "judge_reverse")
            if expected_status == "context_exceeded":
                client.chat.completions.create.assert_not_called()
            else:
                self.assertEqual(trace["response"], '{"choice":2}')
            client.close.assert_called_once()

    def test_pass_or_bad_audit_uses_first_without_retry(self):
        for verdict in ('{"verdict":"PASS","critique":"Fine."}', "malformed"):
            row, events = self.run_job([verdict])
            self.assertEqual([e[0] for e in events], ["run", "audit", "evaluate"])
            self.assertFalse(row["retry"]["triggered"])

    def test_bad_pairwise_result_keeps_first(self):
        row, events = self.run_job(['{"verdict":"RETRY","critique":"Check the recipient."}', '{"choice":"2"}'])
        self.assertEqual(row["retry"]["selected"], 1)
        self.assertEqual(events[-1], ("evaluate", "spine/gate1/a_r1"))

    def test_explicit_fail_uses_retry_without_critic_when_second_completes(self):
        row, events = self.run_job([], first_status="fail")
        self.assertEqual([event[0] for event in events], ["run", "run", "evaluate"])
        self.assertEqual(row["retry"]["selected"], 2)

    def test_runtime_exception_uses_completed_retry_directly(self):
        row, events = self.run_job([], error="RuntimeError: unavailable")
        self.assertEqual([event[0] for event in events], ["run", "run", "evaluate"])
        self.assertIn("unavailable", row["retry"]["critique"])

    def test_default_mode_has_no_audit_metadata_or_calls(self):
        row, events = self.run_job([], retry_mode="none")
        self.assertEqual([event[0] for event in events], ["run", "evaluate"])
        self.assertNotIn("retry", row)
        self.assertNotIn("selected_exp", row)

    def test_attempt_closes_sandbox_and_passes_only_agent_options(self):
        module = ModuleType("appworld")
        world = Mock()
        world.parse_environment_io_log.return_value = []
        module.AppWorld = world
        agent = Mock(guardrail_state=None, step_number=0)
        cfg = {"options": {"retry_mode": "selective", "critic_model": "critic", "guardrails": "full",
                             "critic_thinking": True, "retry_trigger": "risk_only", "retry_temperature": 0.4},
               "agent": {"skip_if_finished": False}, "header": "header"}
        with (
            patch.dict(sys.modules, {"appworld": module}),
            patch.dict(run_lesson._CFG, cfg, clear=True),
            patch.object(run_lesson, "make_agent", return_value=agent) as make,
        ):
            run_lesson._run_attempt("task", "clean_retry", "", "critique")
        self.assertEqual(world.close_all.call_count, 2)
        make.assert_called_once_with(cfg["agent"], "", "header", "critique", guardrails="full")
        agent.solve_tasks.assert_called_once_with(task_ids=["task"], experiment_name="clean_retry")

    def test_selective_worker_disables_reference_data_but_default_worker_does_not(self):
        def config():
            return {"model_config": {"name": "base"}, "appworld_config": {}, "skip_if_finished": True}
        with (
            patch.object(run_lesson, "setup_env"),
            patch.object(run_lesson, "load_agent_config", side_effect=config),
            patch.dict(run_lesson._CFG, {}, clear=True),
        ):
            run_lesson._init("base", 300)
            self.assertTrue(run_lesson._CFG["agent"]["skip_if_finished"])
            self.assertNotIn("load_ground_truth", run_lesson._CFG["agent"]["appworld_config"])
            run_lesson._init("adapter", 300, options={"retry_mode": "selective"})
            self.assertFalse(run_lesson._CFG["agent"]["skip_if_finished"])
            self.assertFalse(run_lesson._CFG["agent"]["appworld_config"]["load_ground_truth"])


class HarnessIntegrationTests(unittest.TestCase):
    def setUp(self):
        # This stand-in exercises the same three hook signatures as pinned AppWorld.
        module_name = "appworld_agents.code.simplified.react_code_agent"
        module = ModuleType(module_name)

        class ReAct:
            def initialize(self, world):
                self.messages = [{"role": "user", "content": "Task: " + world.task.instruction + "\n\n"}]
                self.requests = []

            @property
            def trimmed_messages(self):
                return self.messages

            def next_execution_inputs_usage_and_status(self, outputs):
                if outputs:
                    self.messages.append({"role": "user", "content": "Output:\n" + outputs[0].content})
                self.requests.append(self.trimmed_messages)
                self.messages.append({"role": "assistant", "content": "generated code"})
                return [SimpleNamespace(content=GuardrailTests.submit)], None, None

        module.SimplifiedReActCodeAgent = ReAct
        patcher = patch.dict(sys.modules, {module_name: module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.executed = []
        self.world = SimpleNamespace(
            task=SimpleNamespace(instruction="Find my Spotify tracks"),
            batch_execute=self.execute,
        )

    def execute(self, codes):
        self.executed.extend(codes)
        return ["Observed state"] * len(codes)

    def test_auto_doc_appends_to_same_observation_without_extra_execution(self):
        self.world.apis = SimpleNamespace(api_docs=Mock())
        self.world.apis.api_docs.show_api_doc.return_value = {"parameters": [{"name": "file_path"}]}
        self.world.batch_execute = Mock(return_value=["Execution failed. Traceback:\nTypeError: Unexpected parameter source"])
        native = self.world.batch_execute
        agent = run_lesson.make_agent({}, "", guardrails="full")
        agent.initialize(self.world)
        code = "apis.file_system.move_file(source='x')"
        observation, = self.world.batch_execute([code])
        self.assertIn("Unexpected parameter source", observation)
        self.assertIn("[Harness API Doc]", observation)
        native.assert_called_once_with([code])
        self.assertEqual(agent.guardrail_state.stats()["auto_doc_injections"], 1)

    def test_rsi_core_never_looks_up_or_appends_auto_docs(self):
        self.world.apis = SimpleNamespace(api_docs=Mock())
        self.world.batch_execute = Mock(return_value=[
            "Execution failed. Traceback:\n422 insufficient balance",
        ])
        native = self.world.batch_execute
        agent = run_lesson.make_agent({}, "", guardrails="full", auto_doc="none", rsi_guards="v5")
        agent.initialize(self.world)
        code = "apis.venmo.create_transaction(amount=12)"
        observation, = self.world.batch_execute([code])
        self.assertIn("[RSI Guard]", observation)
        self.assertNotIn("[Harness API Doc]", observation)
        self.world.apis.api_docs.show_api_doc.assert_not_called()
        self.world.apis.api_docs.show_api_descriptions.assert_not_called()
        native.assert_called_once_with([code])
        self.assertEqual(agent.guardrail_state.stats()["auto_doc_injections"], 0)
        self.assertEqual(agent.guardrail_state.rsi_stats()["venmo_card_and_batch_idempotency"], 1)

    def test_cli_combination_forwarding_and_manifest_isolation(self):
        manifests = []
        for auto_doc, rsi in (("none", "v5"), ("full", "v5"), ("full", "none")):
            argv = ["run_lesson.py", "--cond", "isolated", "--tasks", "dev", "--runs", "r1",
                    "--guardrails", "full", "--auto-doc", auto_doc, "--rsi-guards", rsi]
            with (patch.object(sys, "argv", argv), patch.object(run_lesson, "setup_env"),
                  patch.object(run_lesson, "resolve_tasks", return_value=["synthetic"]),
                  patch.object(run_lesson, "_check_run_settings") as check,
                  patch.object(run_lesson.mp, "get_context") as context,
                  redirect_stdout(StringIO())):
                pool = context.return_value.Pool.return_value.__enter__.return_value
                pool.imap_unordered.return_value = [{"success": True}]
                run_lesson.main()
                options = context.return_value.Pool.call_args.kwargs["initargs"][3]
                self.assertEqual(options["auto_doc"], auto_doc)
                self.assertEqual(options["rsi_guards"], rsi)
                settings = check.call_args.args[1]
                self.assertEqual(settings["harness_revision"], 5)
                self.assertEqual(settings["auto_doc"], auto_doc)
                manifests.append(settings)
        with tempfile.TemporaryDirectory() as directory, patch.object(run_lesson, "ROOT", Path(directory)):
            run_lesson._check_run_settings("same_r1", manifests[0])
            for different in manifests[1:]:
                with self.assertRaisesRegex(ValueError, "fresh"):
                    run_lesson._check_run_settings("same_r1", different)

    def test_job_saves_rsi_counters_separately_from_legacy_guardrail_statistics(self):
        evaluator = ModuleType("appworld.evaluator")
        evaluator.evaluate_task = Mock(return_value=SimpleNamespace(to_dict=lambda: {
            "success": True, "failures": [],
        }))
        state = ExecutionGuardrails("full", rsi_guards="v5")
        state.after_execute("apis.phone.search_text_messages(query='invoice')", "[]")
        attempt = {"exp": "spine/gate1/rsi_r1", "steps": [], "error": None,
                   "guardrails": state.stats(), "num_agent_steps": 1,
                   "rsi_triggers": state.rsi_stats(), "rsi_hits": state.rsi_hits}
        with (tempfile.TemporaryDirectory() as directory, patch.object(run_lesson, "ROOT", Path(directory)),
              patch.dict(run_lesson._CFG, {"options": {"guardrails": "full", "rsi_guards": "v5"}}, clear=True),
              patch.dict(sys.modules, {"appworld.evaluator": evaluator}),
              patch.object(run_lesson, "_run_attempt", return_value=attempt)):
            record = run_lesson._job(("rsi", "r1", "synthetic", ""))
        self.assertEqual(len(record["guardrails"]), 6)
        self.assertEqual(record["rsi_triggers"]["search_precision_and_context"], 1)
        self.assertEqual(record["rsi_hits"][0]["key"], "message_context")

    def test_legacy_static_prompt_and_none_mode(self):
        agent = run_lesson.make_agent({}, "Rule", run_lesson.PLAYBOOK_HEADER, "\nRetry")
        native_execute = self.world.batch_execute
        agent.initialize(self.world)
        self.assertEqual(agent.messages[0]["content"],
                         "Task: Find my Spotify tracks" + run_lesson.PLAYBOOK_HEADER + "Rule\n\nRetry")
        self.assertIs(self.world.batch_execute, native_execute)
        none = run_lesson.make_agent({}, "Rule", injection_mode="none")
        none.initialize(self.world)
        self.assertEqual(none.messages[0]["content"], "Task: Find my Spotify tracks\n\n")

    def test_jit_hints_are_request_local_and_router_resets_between_tasks(self):
        agent = run_lesson.make_agent(
            {}, "[track] helpful=1 harmful=0 :: Check Spotify track IDs.", injection_mode="jit_scoped",
        )
        agent.initialize(self.world)
        agent.next_execution_inputs_usage_and_status([])
        self.assertIn("[Scoped Playbook]", agent.requests[0][0]["content"])
        agent.next_execution_inputs_usage_and_status([SimpleNamespace(content="Observed state")])
        self.assertNotIn("[Scoped Playbook]", agent.requests[1][-2]["content"])
        self.assertTrue(all("[Scoped Playbook]" not in m["content"] for m in agent.messages))
        agent.initialize(self.world)
        agent.next_execution_inputs_usage_and_status([])
        self.assertIn("[Scoped Playbook]", agent.requests[0][0]["content"])

    def test_verification_observation_reaches_model_without_a_jit_hint(self):
        agent = run_lesson.make_agent(
            {}, "[error] helpful=1 harmful=0 :: Inspect a Spotify KeyError.", injection_mode="jit_scoped",
        )
        agent.initialize(self.world)
        agent.next_execution_inputs_usage_and_status([])
        marker = "[Harness Verification Required]\nKeyError while preparing Spotify submit"
        agent.next_execution_inputs_usage_and_status([SimpleNamespace(content=marker)])
        last_user = next(m["content"] for m in reversed(agent.requests[1]) if m["role"] == "user")
        self.assertIn(marker, last_user)
        self.assertNotIn("[Scoped Playbook]", last_user)

    def test_query_instruction_is_forwarded_to_submit_gate(self):
        self.world.task.instruction = "How many songs are in the playlist?"
        agent = run_lesson.make_agent({}, "", guardrails="full")
        agent.initialize(self.world)
        code = "apis.supervisor.complete_task(answer=5)"
        self.assertEqual(self.world.batch_execute([code]), ["Observed state"])
        self.assertEqual(self.executed, [code])
        self.assertEqual(agent.guardrail_state.stats()["submit_gate_intercepts"], 0)

    def test_guarded_batch_holds_submit_and_returns_observation(self):
        agent = run_lesson.make_agent({}, "", guardrails="full")
        agent.initialize(self.world)
        self.world.batch_execute([GuardrailTests.write])
        observation, = self.world.batch_execute([GuardrailTests.submit])
        self.assertIn("[Harness Verification Required]", observation)
        self.assertEqual(self.executed, [GuardrailTests.write])
        self.world.batch_execute([GuardrailTests.submit])
        self.assertEqual(self.executed, [GuardrailTests.write, GuardrailTests.submit])
        self.assertEqual(agent.guardrail_state.stats()["readback_intercepts"], 1)
        new_world = SimpleNamespace(task=self.world.task, batch_execute=self.execute)
        agent.initialize(new_world)
        self.assertEqual(agent.guardrail_state.stats(), ExecutionGuardrails().stats())

    def test_opt_in_configuration_cannot_reuse_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_lesson, "ROOT", Path(directory)):
            old_output = Path(directory) / "experiments" / "outputs" / "spine" / "gate1" / "legacy"
            old_output.mkdir(parents=True)
            settings = {"injection_mode": "jit_scoped"}
            with self.assertRaisesRegex(ValueError, "fresh"):
                run_lesson._check_run_settings("legacy", settings)
            run_lesson._check_run_settings("jit", settings)
            run_lesson._check_run_settings("jit", settings)
            with self.assertRaisesRegex(ValueError, "fresh"):
                run_lesson._check_run_settings("jit", None)


if __name__ == "__main__":
    unittest.main()
