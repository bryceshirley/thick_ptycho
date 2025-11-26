import pytest


# verify that cache only updates when n_id changes
# Do n += 0.0 to ensure same array but different id
# Verus n = n + 0.0 which creates a new array


# Replace the following method in thick_ptycho/thick_ptycho/forward_model/base/base_solver.py
# n_id = id(n)
#         if self.projection_cache[proj_idx].modes[mode].cached_n_id != n_id:
#             self.projection_cache[proj_idx].modes[mode].reset(n_id)
#             print(f"Resetting cache for projection {proj_idx}, mode {mode}.")
#             cache_was_reset = True
# with
# old_n = np.copy(self.nk)
# self.nk += gradient_update
# if not np.array_equal(self.nk, old_n):
#     print("nk has changed")
# As it allows for us to do n+=1.0 with triggering cache reset. cache should change when n changes in value, not id.