"""Forward requested label IDs through native speculative logprob extraction.

All probabilities still come from vLLM's compute_topk_scores. Remove this
adaptation automatically when the upstream rejection sampler handles label IDs.
"""
import ast

from .engine_api import UnsupportedEngine, find_symbol


def patch_rejection_source(source):
    text = source.decode()
    method = find_symbol(text, 'RejectionSampler._get_logprobs_tensors')
    if any(isinstance(n, ast.keyword) and n.arg == 'logprob_token_ids_state'
           for n in ast.walk(method)):
        return text
    tree = ast.parse(text)
    changed = set()

    class Patch(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name in ('_get_logprobs_tensors', '_verify_in_chunks'):
                node.args.kwonlyargs.append(ast.arg(arg='max_per_req_token_ids'))
                node.args.kw_defaults.append(None)
                if node.name == '_get_logprobs_tensors':
                    node.args.kwonlyargs.append(ast.arg(arg='expanded_idx_mapping'))
                    node.args.kw_defaults.append(None)
                changed.add(node.name)
            return node

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'max_num_logprobs'
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'self.sampler.sampling_states.max_num_logprobs'):
                changed.add('dimensions')
                return ast.parse('max_num_logprobs, max_per_req_token_ids = '
                    'self.sampler.get_logprobs_dims(input_batch.idx_mapping_np) or (NO_LOGPROBS, 0)').body[0]
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            name = ast.unparse(node.func)
            if name in ('self._get_logprobs_tensors', 'self._verify_in_chunks', 'compute_topk_scores'):
                node.keywords.append(ast.keyword(arg='max_per_req_token_ids',
                                                value=ast.Name(id='max_per_req_token_ids', ctx=ast.Load())))
                changed.add(name)
            if name == 'self._get_logprobs_tensors':
                node.keywords.append(ast.keyword(arg='expanded_idx_mapping',
                    value=ast.parse('input_batch.expanded_idx_mapping[lo:hi]', mode='eval').body))
            if name == 'compute_topk_scores':
                for key, expression in (
                        ('expanded_idx_mapping', 'expanded_idx_mapping'),
                        ('logprob_token_ids_state', 'self.sampler.logprob_token_ids_state')):
                    node.keywords.append(ast.keyword(arg=key, value=ast.parse(expression, mode='eval').body))
            return node

    cls = find_symbol(text, 'RejectionSampler')
    for name in ('_verify_in_chunks', '__call__'):
        find_symbol(text, 'RejectionSampler.' + name)
    # Scope the transformation to this class; preserve unrelated helpers.
    index = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.ClassDef) and node.name == cls.name)
    tree.body[index] = Patch().visit(tree.body[index])
    expected = {'_get_logprobs_tensors', '_verify_in_chunks', 'dimensions',
                'self._get_logprobs_tensors', 'self._verify_in_chunks', 'compute_topk_scores'}
    if changed != expected:
        raise UnsupportedEngine('Native speculative label logprob hooks changed; update the adapter')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'
