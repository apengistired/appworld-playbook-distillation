"""Opt-in, task-independent AST rules distilled from API execution failures."""
from __future__ import annotations

import ast
import json
import re

RSI_MODES = ('none', 'v5')
RSI_RULES = (
    'pagination_limit_guard', 'venmo_card_and_batch_idempotency',
    'search_precision_and_context', 'file_batch_rename_guard',
)
_LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_path(node.value), node.attr)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return (*_path(node.value), node.slice.value)
    return ()


def _names(node):
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _five_items(observation):
    if re.search(r'\b(?:Retrieved|Found)\s+5\b', observation, re.I):
        return True
    try:
        value = json.loads(observation)
    except (ValueError, RecursionError):
        try:
            value = ast.literal_eval(observation.strip())
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return False
    return isinstance(value, (list, tuple)) and len(value) == 5


class _Code:
    def __init__(self, tree, calls):
        self.tree = tree
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and len(_path(node.func)) == 3 and _path(node.func)[0] == 'apis']
        self.kinds = {node: kind for kind, node in calls}
        self.bindings = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bindings[target.id] = node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(node.target, ast.Name):
                self.bindings[node.target.id] = node.value

    def ancestors(self, node):
        while node in self.parents:
            node = self.parents[node]
            yield node

    def loop(self, node):
        return next((parent for parent in self.ancestors(node) if isinstance(parent, _LOOPS)), None)

    def aliases(self, call):
        names = {name for name, value in self.bindings.items() if value is call}
        for _ in range(len(self.bindings) + 1):
            previous = set(names)
            for name, value in self.bindings.items():
                if value is not None and _names(value) & names:
                    names.add(name)
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and _names(node.iter) & names:
                    names.update(_names(node.target))
            if names == previous:
                break
        return names

    def string(self, node, seen=None):
        seen = set() if seen is None else seen
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id not in seen and node.id in self.bindings:
            return self.string(self.bindings[node.id], seen | {node.id})
        return None


def _len_of(node, aliases, call):
    return (isinstance(node, ast.Call) and _path(node.func) == ('len',) and len(node.args) == 1
            and (node.args[0] is call or bool(_names(node.args[0]) & aliases)))


def _bad_while_limit(context, call):
    loop = context.loop(call)
    if not isinstance(loop, ast.While):
        return False
    aliases = context.aliases(call)
    for condition in (node for node in ast.walk(loop) if isinstance(node, ast.If)):
        # A break in an inner loop does not terminate this pagination loop.
        if context.loop(condition) is not loop or not any(
            isinstance(node, ast.Break) and context.loop(node) is loop for node in ast.walk(condition)
        ):
            continue
        for compare in (node for node in ast.walk(condition.test) if isinstance(node, ast.Compare)):
            for left, op, right in zip([compare.left, *compare.comparators[:-1]], compare.ops, compare.comparators):
                if (_len_of(left, aliases, call) and isinstance(op, (ast.Lt, ast.LtE, ast.NotEq))
                        and isinstance(right, ast.Constant) and type(right.value) is int and right.value > 5):
                    return True
                if (_len_of(right, aliases, call) and isinstance(op, (ast.Gt, ast.GtE, ast.NotEq))
                        and isinstance(left, ast.Constant) and type(left.value) is int and left.value > 5):
                    return True
    # A continuation condition can terminate the loop just as an explicit break can.
    test = loop.test
    return (isinstance(test, ast.Compare) and len(test.ops) == 1
            and _len_of(test.left, aliases, call) and isinstance(test.ops[0], (ast.GtE, ast.Gt, ast.Eq))
            and isinstance(test.comparators[0], ast.Constant)
            and type(test.comparators[0].value) is int and test.comparators[0].value > 5)


def _per_item_error_handled(context, call):
    loop = context.loop(call)
    if loop is None:
        return False
    for parent in context.ancestors(call):
        if parent is loop:
            break
        if not isinstance(parent, ast.Try):
            continue
        for handler in parent.handlers:
            broad = handler.type is None or _path(handler.type) in (('Exception',), ('BaseException',))
            if broad and not any(isinstance(node, (ast.Raise, ast.Return, ast.Break)) for node in ast.walk(handler)):
                return True
            if broad and any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                             and 'already' in node.value.lower() for node in ast.walk(handler)) and any(
                    isinstance(node, ast.Continue) for node in ast.walk(handler)):
                return True
    return False


def _expanded_nodes(node, context):
    pending = [node]
    expanded = set()
    while pending:
        item = pending.pop()
        yield item
        pending.extend(ast.iter_child_nodes(item))
        if isinstance(item, ast.Name) and item.id in context.bindings and item.id not in expanded:
            expanded.add(item.id)
            if context.bindings[item.id] is not None:
                pending.append(context.bindings[item.id])


def _has_exact_user_match(context, aliases):
    for compare in (node for node in ast.walk(context.tree) if isinstance(node, ast.Compare)):
        if not any(isinstance(op, ast.Eq) for op in compare.ops):
            continue
        for node in _expanded_nodes(compare, context):
            field = None
            source = None
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                field, source = node.slice.value, node.value
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and node.args:
                field = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
                source = node.func.value
            elif isinstance(node, ast.Attribute):
                field, source = node.attr, node.value
            if field in ('email', 'name', 'full_name', 'first_name', 'last_name') and source is not None and _names(source) & aliases:
                return True
    return False


def _uses_today(node, context, seen=None):
    seen = set() if seen is None else seen
    if node is None:
        return False
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and _path(item.func)[-1:] == ('now',):
            return True
        if isinstance(item, ast.Name):
            if item.id in seen:
                continue
            if item.id in context.bindings:
                if _uses_today(context.bindings[item.id], context, seen | {item.id}):
                    return True
            elif item.id == 'current_date':
                return True
    return False


def _already_filters_prefix(context):
    for condition in (node for node in ast.walk(context.tree) if isinstance(node, ast.If)):
        if not any(isinstance(node, ast.Continue) for node in ast.walk(condition)):
            continue
        for call in (node for node in ast.walk(condition.test) if isinstance(node, ast.Call)):
            if _path(call.func)[-1:] == ('startswith',) and call.args:
                arg = call.args[0]
                if any('prefix' in name.lower() for name in _names(arg)):
                    return True
                text = context.string(arg)
                if text and re.match(r'^\d{2,4}[-_]', text):
                    return True
            if _path(call.func)[-1:] in (('match',), ('search',)) and call.args:
                text = context.string(call.args[0])
                if text and r'\d{' in text and text.startswith('^'):
                    return True
    return False


class RSICodeGuards:
    """Four deterministic rule families, deduplicated by API/subrule within a task."""
    def __init__(self, mode='none'):
        if mode not in RSI_MODES:
            raise ValueError(f'Unknown RSI guard mode: {mode}')
        self.mode = mode
        self.counts = dict.fromkeys(RSI_RULES, 0)
        self.hits = []
        self.seen = set()
        self.pagination_apis = set()
        self.playlist_seen = False
        self.song_details_seen = False
        self.move_count = 0
        self.move_failed = False
        self.step = 0

    def emit(self, rule, key, message, messages):
        token = (rule, key)
        if token in self.seen:
            return
        self.seen.add(token)
        self.counts[rule] += 1
        self.hits.append({'step': self.step, 'rule': rule, 'key': key})
        messages.append(message)

    def pagination_limit_guard(self, context, observation, messages):
        for call in context.calls:
            path = _path(call.func)
            if context.kinds.get(call) != 'read' or path[1] in ('api_docs', 'supervisor'):
                continue
            keywords = {kw.arg for kw in call.keywords}
            if 'page_limit' in keywords or None in keywords:
                continue
            wrong_loop = 'page_index' in keywords and _bad_while_limit(context, call)
            # Only plausible paged APIs qualify for the count-only branch. Fixed
            # enumerations and complete queues must not be guessed to be paginated.
            plausible_paged = ('page_index' in keywords or path[-1].startswith(('search_', 'list_'))
                               or path[-1].endswith('_library') or path[-1] in (
                                   'show_social_feed', 'show_sent_payment_requests', 'show_received_payment_requests'))
            five = context.loop(call) is None and plausible_paged and _five_items(observation)
            if wrong_loop or five:
                name = '.'.join(path)
                self.pagination_apis.add(name)
                self.emit(RSI_RULES[0], 'limit:' + name,
                          'List/search APIs default to page_limit=5, not 20. Without page_limit=20, len(page) < 20 '
                          'can stop on page 0. Set page_limit=20 explicitly, or stop only when the page is empty.', messages)
        self.playlist_seen |= any(_path(call.func) == ('apis', 'spotify', 'show_playlist') for call in context.calls)
        self.song_details_seen |= any(_path(call.func) == ('apis', 'spotify', 'show_song') for call in context.calls)
        rating = any((isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 'rating')
                     or (isinstance(node, ast.Attribute) and node.attr == 'rating')
                     or (isinstance(node, ast.Call) and _path(node.func)[-1:] == ('get',) and node.args
                         and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'rating')
                     for node in ast.walk(context.tree))
        if self.playlist_seen and rating and not self.song_details_seen:
            self.emit(RSI_RULES[0], 'playlist_rating',
                      'show_playlist entries may omit song ratings; fetch show_song(song_id) for the real rating.', messages)

    def venmo_card_and_batch_idempotency(self, context, observation, messages):
        if not re.search(r'\b422\b', observation):
            return
        create = any(_path(call.func) == ('apis', 'venmo', 'create_transaction') for call in context.calls)
        if create and re.search(r'insufficient\s+(?:\w+\s+)?balance|balance\s+does\s+not\s+have|not enough (?:balance|funds)', observation, re.I):
            self.emit(RSI_RULES[1], 'balance_card',
                      'Unless a top-up is requested, use show_payment_cards() to choose an unexpired card and pass '
                      'payment_card_id directly to create_transaction; avoid add_to_venmo_balance and extra side effects.', messages)
        if re.search(r'\balready\b', observation, re.I):
            for call in context.calls:
                if context.kinds.get(call) == 'write' and context.loop(call) and not _per_item_error_handled(context, call):
                    self.emit(RSI_RULES[1], 'batch:' + '.'.join(_path(call.func)),
                              'Handle already-existing state with per-item try/except and continue the batch; '
                              'verify the full target set before complete_task().', messages)

    def search_precision_and_context(self, context, observation, messages):
        for call in context.calls:
            path = _path(call.func)
            if path == ('apis', 'venmo', 'search_users'):
                aliases = context.aliases(call)
                checked = any((bool(_names(node.test) & aliases) or call in tuple(ast.walk(node.test)))
                              for node in ast.walk(context.tree) if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)))
                if checked and not _has_exact_user_match(context, aliases):
                    self.emit(RSI_RULES[2], 'fuzzy_users',
                              'venmo.search_users is fuzzy. Check each returned email or full name for an exact match; '
                              'a nonempty result does not establish that the requested account exists.', messages)
            if path == ('apis', 'phone', 'search_text_messages'):
                query = next((kw.value for kw in call.keywords if kw.arg == 'query'), None)
                text = context.string(query)
                if text and text.strip():
                    self.emit(RSI_RULES[2], 'message_context',
                              'A text-message query can omit replies containing only numbers. Fetch the complete '
                              'conversation by phone_number without query, then inspect it in time order.', messages)

    def file_batch_rename_guard(self, context, observation, messages):
        moves = [call for call in context.calls if _path(call.func) == ('apis', 'file_system', 'move_file')]
        batch = len(moves) > 1 or any(context.loop(call) is not None for call in moves)
        for call in moves:
            destination = next((kw.value for kw in call.keywords if kw.arg and
                                (kw.arg.startswith(('destination', 'target')) or kw.arg in ('new_path', 'dest'))), None)
            if destination is None and len(call.args) >= 2:
                destination = call.args[-1]
            if batch and _uses_today(destination, context):
                self.emit(RSI_RULES[3], 'creation_date',
                          'For date prefixes, check whether the task requires each file\'s created_at rather than datetime.now()/current_date.', messages)
        failed = bool(moves and re.search(r'Execution failed|Traceback|\w*Error:|\w*Exception:', observation, re.I))
        relevant = bool(moves or any(_path(call.func) == ('apis', 'file_system', 'show_directory') for call in context.calls))
        repeated = self.move_failed or failed or self.move_count + len(moves) > 1 or batch
        if relevant and repeated and not _already_filters_prefix(context):
            self.emit(RSI_RULES[3], 'rename_idempotency',
                      'Before retrying a rename batch, re-list the directory and skip names already carrying '
                      'the target date prefix to prevent double prefixes.', messages)
        self.move_failed |= failed
        self.move_count += len(moves) + int(batch)

    def after_execute(self, tree, calls, observation):
        if self.mode == 'none':
            return None
        self.step += 1
        context = _Code(tree, calls)
        messages = []
        for rule in (self.pagination_limit_guard, self.venmo_card_and_batch_idempotency,
                     self.search_precision_and_context, self.file_batch_rename_guard):
            rule(context, observation, messages)
        return '[RSI Guard]\n' + '\n'.join('- ' + message for message in messages) if messages else None
