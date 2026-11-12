import os
import json

def audit_regime_feature_overlap(profile_dir: str = "data/regime_profiles"):
    """
    Loads all regime JSON profiles and computes exact feature intersections
    across core features, satellite pools, and total combined feature sets.
    """
    profile_files = [f for f in os.listdir(profile_dir) if f.endswith(".json")]
    
    if not profile_files:
        print(f"⚠️ No profile JSON files found in {profile_dir}")
        return
        
    regime_data = {}
    for file in profile_files:
        path = os.path.join(profile_dir, file)
        with open(path, "r") as f:
            profile = json.load(f)
            regime_name = profile.get("regime", file)
            # Combine core and satellite into a full set for this regime
            all_regime_features = set(profile.get("core_features", [])) | set(profile.get("satellite_pool", []))
            regime_data[regime_name] = {
                "core": set(profile.get("core_features", [])),
                "all": all_regime_features
            }
            
    # Compute Intersections
    regime_names = list(regime_data.keys())
    
    # Start with the first regime's sets
    universal_cores = regime_data[regime_names[0]]["core"]
    universal_all = regime_data[regime_names[0]]["all"]
    
    for name in regime_names[1:]:
        universal_cores = universal_cores.intersection(regime_data[name]["core"])
        universal_all = universal_all.intersection(regime_data[name]["all"])
        
    print("=" * 60)
    print("🔍 REGIME FEATURE OVERLAP AUDIT")
    print("=" * 60)
    print(f"Analyzed Regimes: {regime_names}\n")
    
    print("📌 Universal Core Features (Present in Core across ALL regimes):")
    if universal_cores:
        for f in universal_cores:
            print(f"   - {f}")
    else:
        print("   (None! No single feature made the core cut in every regime.)")
        
    print("\n📌 Universal Total Features (Present anywhere across ALL regimes):")
    if universal_all:
        for f in universal_all:
            print(f"   - {f}")
    else:
        print("   (None! Every regime relies on mutually exclusive feature pools.)")
    print("=" * 60)
    
    return universal_cores, universal_all

if __name__ == '__main__':
    universal_cores, universal_all = audit_regime_feature_overlap()
    print(f" Universal Cores: {universal_cores}")
    print(f" Universal All : {universal_all}")