import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def run_cli(self,*args,env=None):
        return subprocess.run([sys.executable,'-m','agentrx',*map(str,args)],cwd=ROOT,
                              text=True,capture_output=True,env=env)

    def test_offline_smoke_exercises_pipeline_and_three_repetitions(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_cli('smoke','--out',d)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((Path(d)/'experiment/summary.json').read_text())
            self.assertEqual(summary['measurement_kind'],'offline_fixture')
            self.assertEqual(len(summary['results'][0]['repetitions']),3)
            self.assertIn('NOT BENCHMARK', (Path(d)/'experiment/table7.md').read_text())
            jobs=list((Path(d)/'experiment/jobs').glob('**/result.json'))
            combined=[json.loads(p.read_text()) for p in jobs if 'agentrx' in str(p)]
            self.assertEqual(len(combined),3)
            self.assertEqual(combined[0]['prediction']['step_index'],3)
            self.assertEqual(combined[0]['prediction']['category'],'MisinterpretationOfToolOutput')
            self.assertTrue(any(c['verdict']=='VIOL' for c in combined[0]['checks']))

    def test_missing_credentials_fail_before_run_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_cli('smoke','--out',Path(d)/'smoke').returncode,0)
            env={k:v for k,v in os.environ.items() if k not in ('OPENAI_API_KEY','AZURE_OPENAI_API_KEY')}
            result=self.run_cli('run','--corpus',Path(d)/'smoke/corpus',
                                '--config','configs/pilot_gpt5.json','--out',Path(d)/'live',env=env)
            self.assertEqual(result.returncode,2)
            self.assertIn('OPENAI_API_KEY',result.stderr)
            self.assertFalse((Path(d)/'live/experiment.json').exists())

    def test_stress_is_unlabeled_and_reports_no_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_cli('stress','--input','examples/count_mismatch.json',
                                '--config','examples/stress_config.json','--out',d)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((Path(d)/'stress_summary.json').read_text())
            self.assertEqual(summary['input_count'],1)
            self.assertIsNone(summary['gold_accuracy'])

    def test_model_flag_routing_and_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_cli('smoke','--out',Path(d)/'smoke').returncode,0)
            res_ollama = self.run_cli('run','--corpus',Path(d)/'smoke/corpus',
                                      '--model','ollama','--preflight','--out',Path(d)/'ollama')
            self.assertEqual(res_ollama.returncode,0,res_ollama.stderr)
            info = json.loads(res_ollama.stdout)
            self.assertEqual(info['provider']['provider'],'ollama')

            env_no_gemini = {k:v for k,v in os.environ.items() if k != 'GEMINI_API_KEY'}
            res_gem_fail = self.run_cli('run','--corpus',Path(d)/'smoke/corpus',
                                        '--model','gemini','--preflight','--out',Path(d)/'gem',env=env_no_gemini)
            self.assertEqual(res_gem_fail.returncode,2)
            self.assertIn('GEMINI_API_KEY',res_gem_fail.stderr)

            env_gemini = {**env_no_gemini, 'GEMINI_API_KEY': 'test-gem-key'}
            res_gem_ok = self.run_cli('run','--corpus',Path(d)/'smoke/corpus',
                                      '--model','gemini','--preflight','--out',Path(d)/'gem',env=env_gemini)
            self.assertEqual(res_gem_ok.returncode,0,res_gem_ok.stderr)
            gem_info = json.loads(res_gem_ok.stdout)
            self.assertEqual(gem_info['provider']['provider'],'gemini')

