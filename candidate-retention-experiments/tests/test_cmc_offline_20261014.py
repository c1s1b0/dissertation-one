"""新增离线支持度排序和审计分母的独立行为核验。"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('cmc_offline_20261014'), '缺少已批准的离线分析模块')
        import cmc_offline_20261014
        self.m=cmc_offline_20261014

    def test_support_uses_compatibility_and_preserves_initial_role_quota(self):
        # 若忽略二元支持度将按 ID 误选 a1；若删去初始池或按总数匹配也会失败。
        t=NS(required_roles=('a','b'),candidate_ids={'a':['a0'],'b':['b0']},numeric_requirements={})
        cs=[NS(template_id=n,role=n[0],parameters={}) for n in ('a0','a1','a2','b0','b1')]
        def compat(a,b):return a.template_id!='a1' and b.template_id!='a1'
        ids,meta=self.m.support_pool(t,cs,['a0','a2','b0'],[('a','b')],compat)
        self.assertEqual(set(ids),{'a0','a2','b0'})
        self.assertEqual(meta['pair_checks'],12)
        with self.assertRaises(ValueError):self.m.support_pool(t,cs,['a2','b0'],[('a','b')],compat)

    def test_unary_invalid_candidate_cannot_outrank_valid(self):
        t=NS(required_roles=('a','b'),candidate_ids={'a':['a0'],'b':['b0']},numeric_requirements={'a.x.min':5})
        cs=[NS(template_id=n,role=n[0],parameters={'x':v}) for n,v in [('a0',0),('a1',4),('a2',6),('b0',0)]]
        ids,_=self.m.support_pool(t,cs,['a0','a1','b0'],[('a','b')],lambda a,b:a.template_id!='a2' and b.template_id!='a2')
        self.assertEqual(set(ids),{'a0','a2','b0'})

    def test_pair_predicate_matches_known_numeric_direction_and_ports(self):
        port=(NS(direction='in',port_type='m',unit='u'),NS(direction='out',port_type='m',unit='u'))
        t=NS(scene='B',required_roles=('motor','gearbox','coupling','shaft_support'))
        a=NS(role='motor',parameters={'speed_rpm':100,'torque_nm':10,'shaft_diameter_mm':5},ports=port)
        b=NS(role='gearbox',parameters={'input_speed_rpm':110},ports=port)
        self.assertFalse(self.m.pair_compatible(t,a,b))
        self.assertFalse(self.m.pair_compatible(t,b,a))
        b.parameters['input_speed_rpm']=90
        self.assertTrue(self.m.pair_compatible(t,a,b))
        b.ports=()
        self.assertFalse(self.m.pair_compatible(t,a,b))

    def test_audit_missing_usage_not_zero_and_retries_not_new_tasks(self):
        base={'task_id':'t','repeat':1,'stage':'initial','task_attempt':1,'latency_ms':10}
        a=[dict(base,status='failed',http_status=None,attempt=1,retry_index=0,usage=None,error_type='URLError'),
           dict(base,status='success',http_status=200,attempt=2,retry_index=1,usage={'prompt_tokens':10,'completion_tokens':2,'total_tokens':12},error_type=None)]
        r=self.m.summarize_audit(a)
        self.assertEqual(r['request_attempts'],2)
        self.assertEqual(r['task_repeats'],1)
        self.assertEqual(r['retried_task_repeats'],1)
        self.assertEqual(r['non200_or_missing_attempts'],1)
        self.assertEqual(r['accepted_proposals'],1)
        self.assertEqual(r['http200_attempts'],1)
        self.assertEqual(r['positive_usage_attempts'],1)
        self.assertEqual(r['reported_total_tokens'],12)
        self.assertIsNone(r['total_billed_tokens'])
        self.assertIsNone(r['monetary_cost'])

    def test_audit_contract_failure_is_not_transport_failure(self):
        base={'task_id':'t','repeat':1,'stage':'initial','task_attempt':1,'latency_ms':1,'attempt':1,'retry_index':0}
        r=self.m.summarize_audit([dict(base,status='failed',http_status=200,error_type='ValidationError',usage={'total_tokens':0})])
        self.assertEqual(r['non200_or_missing_attempts'],0)
        self.assertEqual(r['rejected_http200'],1)
        self.assertEqual(r['positive_usage_attempts'],0)
        self.assertEqual(r['zero_usage_attempts'],1)

    def test_validation_failure_without_http_status_remains_visible(self):
        r=self.m.summarize_audit([dict(task_id='t',repeat=1,stage='initial',task_attempt=1,latency_ms=1,
             attempt=1,retry_index=0,status='failed',http_status=None,error_type='ValidationError',usage=None)])
        self.assertEqual(r['validation_error_records'],1)
        self.assertEqual(r['validation_outcomes'],1)
        self.assertEqual(r['audit_acceptance_rate'],0)

    def test_replay_comparison_ignores_only_local_time_not_results(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_cmc_bundle_20261014'))
        from verify_cmc_bundle_20261014 import comparable
        self.assertEqual(comparable({'feasible':1,'solver_ms':2}),{'feasible':1})
        self.assertNotEqual(comparable({'feasible':0,'solver_ms':2}),comparable({'feasible':1,'solver_ms':3}))
        self.assertEqual(comparable({'final_elapsed_seconds_quantiles':[1,2,3]}),{'final_elapsed_seconds_quantiles':[1,2,3]})

if __name__=='__main__':unittest.main()
