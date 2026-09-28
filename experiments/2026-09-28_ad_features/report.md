# Known-state all-distinct feature screen

Prediction target is the normalized AD error on previously trained FIT networks. Leave-one-network-out fitting below is diagnostic, not independent final validation. Basis projection energies use oracle coefficients, whereas Gram fits use only other networks. No downstream score benefit follows from residual prediction alone.

## Consumer layer 3

| target | feature family | LONO residual energy removed |
|---|---|---:|
| E | local | 0.0001 |
| E | all | 0.0004 |
| Eorth | local | 0.0001 |
| Eorth | all | 0.0004 |

| known basis | rank | median right capture | median left capture | median bilateral capture |
|---|---:|---:|---:|---:|
| covariance | 8 | 0.0927 | 0.0080 | 0.0008 |
| covariance | 16 | 0.1569 | 0.0157 | 0.0025 |
| covariance | 32 | 0.2585 | 0.0313 | 0.0081 |
| covariance | 64 | 0.4025 | 0.0628 | 0.0250 |
| covariance | 128 | 0.5822 | 0.1247 | 0.0729 |
| quadratic_left_covariance_right | 8 | 0.0927 | 0.0120 | 0.0019 |
| quadratic_left_covariance_right | 16 | 0.1569 | 0.0223 | 0.0043 |
| quadratic_left_covariance_right | 32 | 0.2585 | 0.0398 | 0.0107 |
| quadratic_left_covariance_right | 64 | 0.4025 | 0.0751 | 0.0302 |
| quadratic_left_covariance_right | 128 | 0.5822 | 0.1433 | 0.0823 |
## Consumer layer 6

| target | feature family | LONO residual energy removed |
|---|---|---:|
| E | local | 0.0303 |
| E | all | 0.0404 |
| Eorth | local | 0.0242 |
| Eorth | all | 0.0329 |

| known basis | rank | median right capture | median left capture | median bilateral capture |
|---|---:|---:|---:|---:|
| covariance | 8 | 0.2423 | 0.0080 | 0.0022 |
| covariance | 16 | 0.3376 | 0.0159 | 0.0053 |
| covariance | 32 | 0.4594 | 0.0312 | 0.0142 |
| covariance | 64 | 0.5966 | 0.0628 | 0.0379 |
| covariance | 128 | 0.7383 | 0.1254 | 0.0922 |
| quadratic_left_covariance_right | 8 | 0.2423 | 0.0543 | 0.0287 |
| quadratic_left_covariance_right | 16 | 0.3376 | 0.0843 | 0.0452 |
| quadratic_left_covariance_right | 32 | 0.4594 | 0.1094 | 0.0630 |
| quadratic_left_covariance_right | 64 | 0.5966 | 0.1506 | 0.0977 |
| quadratic_left_covariance_right | 128 | 0.7383 | 0.2253 | 0.1676 |
## Consumer layer 9

| target | feature family | LONO residual energy removed |
|---|---|---:|
| E | local | 0.0750 |
| E | all | 0.0967 |
| Eorth | local | 0.0497 |
| Eorth | all | 0.0618 |

| known basis | rank | median right capture | median left capture | median bilateral capture |
|---|---:|---:|---:|---:|
| covariance | 8 | 0.3507 | 0.0092 | 0.0038 |
| covariance | 16 | 0.4561 | 0.0162 | 0.0076 |
| covariance | 32 | 0.5777 | 0.0308 | 0.0175 |
| covariance | 64 | 0.7015 | 0.0630 | 0.0445 |
| covariance | 128 | 0.8182 | 0.1239 | 0.1019 |
| quadratic_left_covariance_right | 8 | 0.3507 | 0.1064 | 0.0633 |
| quadratic_left_covariance_right | 16 | 0.4561 | 0.1518 | 0.0960 |
| quadratic_left_covariance_right | 32 | 0.5777 | 0.1836 | 0.1255 |
| quadratic_left_covariance_right | 64 | 0.7015 | 0.2267 | 0.1678 |
| quadratic_left_covariance_right | 128 | 0.8182 | 0.3013 | 0.2481 |
## Consumer layer 12

| target | feature family | LONO residual energy removed |
|---|---|---:|
| E | local | 0.0976 |
| E | all | 0.1354 |
| Eorth | local | 0.0470 |
| Eorth | all | 0.0765 |

| known basis | rank | median right capture | median left capture | median bilateral capture |
|---|---:|---:|---:|---:|
| covariance | 8 | 0.4031 | 0.0084 | 0.0032 |
| covariance | 16 | 0.5221 | 0.0154 | 0.0076 |
| covariance | 32 | 0.6437 | 0.0308 | 0.0200 |
| covariance | 64 | 0.7564 | 0.0594 | 0.0449 |
| covariance | 128 | 0.8600 | 0.1208 | 0.1039 |
| quadratic_left_covariance_right | 8 | 0.4031 | 0.1239 | 0.0710 |
| quadratic_left_covariance_right | 16 | 0.5221 | 0.1808 | 0.1138 |
| quadratic_left_covariance_right | 32 | 0.6437 | 0.2146 | 0.1506 |
| quadratic_left_covariance_right | 64 | 0.7564 | 0.2655 | 0.2045 |
| quadratic_left_covariance_right | 128 | 0.8600 | 0.3463 | 0.2974 |
## Consumer layer 15

| target | feature family | LONO residual energy removed |
|---|---|---:|
| E | local | 0.1054 |
| E | all | 0.1611 |
| Eorth | local | 0.0368 |
| Eorth | all | 0.0785 |

| known basis | rank | median right capture | median left capture | median bilateral capture |
|---|---:|---:|---:|---:|
| covariance | 8 | 0.4435 | 0.0084 | 0.0040 |
| covariance | 16 | 0.5611 | 0.0164 | 0.0096 |
| covariance | 32 | 0.6803 | 0.0309 | 0.0212 |
| covariance | 64 | 0.7896 | 0.0615 | 0.0490 |
| covariance | 128 | 0.8843 | 0.1239 | 0.1098 |
| quadratic_left_covariance_right | 8 | 0.4435 | 0.1418 | 0.0860 |
| quadratic_left_covariance_right | 16 | 0.5611 | 0.1992 | 0.1275 |
| quadratic_left_covariance_right | 32 | 0.6803 | 0.2402 | 0.1715 |
| quadratic_left_covariance_right | 64 | 0.7896 | 0.2910 | 0.2304 |
| quadratic_left_covariance_right | 128 | 0.8843 | 0.3750 | 0.3272 |

Final-layer reference floor approximately 7.7e-11; moment-target uncertainty is distinct. Current probe baseline final MSE per net: {0: 1.8987437865976184e-08, 1: 1.9953500720243484e-08, 2: 2.2336731841186964e-08, 3: 1.9527684907313694e-08, 4: 1.8444049079984185e-08, 5: 2.3409026707239427e-08, 6: 2.0088315669394503e-08, 7: 2.1695518740545146e-08}