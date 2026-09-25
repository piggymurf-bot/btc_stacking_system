import pandas as pd

from src.config_v2 import FEATURE_GROUPS


def audit_feature_groups(df: pd.DataFrame, feature_groups: dict):
    """
    Systematically checks if any features listed in FEATURE_GROUPS 
    are missing from the dataframe columns.
    """
    df_cols = set(df.columns)
    all_missing = {}

    print("🔍 Running Feature Group Validation Audit...\n")
    for group_name, features in feature_groups.items():
        group_cols = set(features)
        missing = group_cols - df_cols
        
        if missing:
            all_missing[group_name] = list(missing)
            print(f"❌ [{group_name}] MISSING {len(missing)} features:")
            for m in sorted(missing):
                print(f"    - {m}")
        else:
            print(f"✅ [{group_name}] All {len(group_cols)} features present!")

    return all_missing

def check_missing_features():
    df = pd.read_csv("data/processed/master_feature_addlhar_alpha.csv")
    audit_feature_groups(df, FEATURE_GROUPS)
    

if __name__ == "__main__":
    check_missing_features()

