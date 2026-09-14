"""Structural-zero diagnostics and precision boundaries for the E14 remedy."""
import unittest
import p2_compact_mixed_qr_e14 as mixed
import flopscope as flops
import flopscope.numpy as fnp


class TestMixedQR(unittest.TestCase):
    def test_precision_boundary_and_payload(self):
        frozen=mixed.f32.load_reference(mixed.f32.b1.FROZEN,False)
        with flops.BudgetContext(flop_budget=2**25,quiet=True):
            n=4
            eye=fnp.eye(n,dtype=fnp.float32)
            tower=frozen._Tower(eye,fnp.ones((n,n),dtype=fnp.float32)-eye,None,
                                fnp.ones(n,dtype=fnp.float32),1)
            state=(fnp.zeros(n,dtype=fnp.float32),eye,tower,0.0)
            encoded=mixed.encode64(state)
            self.assertTrue(all(r['dtype']=='float64' for r in mixed.payload_inventory(encoded[2])))
            decoded,_=mixed.decode32(frozen,encoded)
            for field in ('A_D','A_L','w2b'):
                self.assertEqual(str(getattr(decoded[2],field).dtype),'float32')
            self.assertIsNone(decoded[2].M)
            self.assertIs(decoded[0],state[0])
            self.assertIs(decoded[1],state[1])

    def test_structural_zero_and_signed_zero_are_separate(self):
        with flops.BudgetContext(flop_budget=2**20,quiet=True):
            zero=fnp.array([0.0],dtype=fnp.float32)
            tiny=fnp.array([1e-30],dtype=fnp.float32)
            signed=fnp.array([-0.0],dtype=fnp.float32)
            changed=mixed.exact_field('zero',tiny,zero)
            sign=mixed.exact_field('sign',signed,zero)
        self.assertFalse(changed['passed'])
        self.assertEqual(changed['zero_mismatches'],1)
        self.assertFalse(sign['passed'])
        self.assertEqual(sign['bit_mismatches'],1)
        self.assertEqual(sign['zero_mismatches'],0)


if __name__=='__main__':unittest.main()
