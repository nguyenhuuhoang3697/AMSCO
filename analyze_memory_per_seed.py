#!/usr/bin/env python3
"""
Script phân tích RAM usage giữa các seeds
"""

import re
import pandas as pd

def parse_memory_from_log(log_file):
    """Parse memory usage từ output.log"""
    
    memory_data = []
    current_seed = None
    current_dataset = None
    current_model = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Detect seed
            if match := re.search(r'SEED (\d+)', line):
                current_seed = int(match.group(1))
            
            # Detect dataset
            if match := re.search(r'ĐANG THỬ NGHIỆM TRÊN BỘ DỮ LIỆU: (\w+)', line):
                current_dataset = match.group(1)
            
            # Detect model
            if match := re.search(r'Đang tối ưu mô hình: (\w+)', line):
                current_model = match.group(1)
            
            # Parse memory line (both Optuna and AMSCO)
            if 'Optuna (TPE)' in line or 'AMSCO' in line:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        optimizer = 'Optuna' if 'Optuna' in line else 'AMSCO'
                        peak_mem = float(parts[-5])
                        rss_mem = float(parts[-4])
                        
                        memory_data.append({
                            'Seed': current_seed,
                            'Dataset': current_dataset,
                            'Model': current_model,
                            'Optimizer': optimizer,
                            'Peak Memory (MB)': peak_mem,
                            'RSS Memory (MB)': rss_mem
                        })
                    except (ValueError, IndexError):
                        pass
    
    return pd.DataFrame(memory_data)

def compare_seeds(df, seed1, seed2):
    """So sánh memory giữa 2 seeds"""
    
    print(f"\n{'='*100}")
    print(f"SO SÁNH RAM USAGE: SEED {seed1} vs SEED {seed2}")
    print(f"{'='*100}\n")
    
    seed1_data = df[df['Seed'] == seed1]
    seed2_data = df[df['Seed'] == seed2]
    
    comparison = []
    
    for _, row1 in seed1_data.iterrows():
        # Find matching row in seed2
        row2 = seed2_data[
            (seed2_data['Dataset'] == row1['Dataset']) &
            (seed2_data['Model'] == row1['Model']) &
            (seed2_data['Optimizer'] == row1['Optimizer'])
        ]
        
        if not row2.empty:
            row2 = row2.iloc[0]
            
            peak_change = row2['Peak Memory (MB)'] - row1['Peak Memory (MB)']
            peak_pct = (peak_change / row1['Peak Memory (MB)']) * 100 if row1['Peak Memory (MB)'] > 0 else 0
            
            rss_change = row2['RSS Memory (MB)'] - row1['RSS Memory (MB)']
            rss_pct = (rss_change / row1['RSS Memory (MB)']) * 100 if row1['RSS Memory (MB)'] > 0 else 0
            
            comparison.append({
                'Dataset': row1['Dataset'],
                'Model': row1['Model'],
                'Optimizer': row1['Optimizer'],
                f'Peak Seed{seed1}': f"{row1['Peak Memory (MB)']:.2f}",
                f'Peak Seed{seed2}': f"{row2['Peak Memory (MB)']:.2f}",
                'Peak Δ (MB)': f"{peak_change:+.2f}",
                'Peak Δ (%)': f"{peak_pct:+.1f}%",
                f'RSS Seed{seed1}': f"{row1['RSS Memory (MB)']:.2f}",
                f'RSS Seed{seed2}': f"{row2['RSS Memory (MB)']:.2f}",
                'RSS Δ (MB)': f"{rss_change:+.2f}",
                'RSS Δ (%)': f"{rss_pct:+.1f}%",
            })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*100}")
    print("TÓM TẮT THAY ĐỔI RAM")
    print(f"{'='*100}\n")
    
    # Extract numeric values for statistics
    peak_changes = [float(x.replace('+', '')) for x in comp_df['Peak Δ (MB)']]
    rss_changes = [float(x.replace('+', '')) for x in comp_df['RSS Δ (MB)']]
    
    print(f"Peak Memory Changes:")
    print(f"  - Trung bình: {sum(peak_changes)/len(peak_changes):+.2f} MB")
    print(f"  - Min: {min(peak_changes):+.2f} MB")
    print(f"  - Max: {max(peak_changes):+.2f} MB")
    print(f"  - Số lần giảm: {sum(1 for x in peak_changes if x < 0)}/{len(peak_changes)}")
    
    print(f"\nRSS Memory Changes:")
    print(f"  - Trung bình: {sum(rss_changes)/len(rss_changes):+.2f} MB")
    print(f"  - Min: {min(rss_changes):+.2f} MB")
    print(f"  - Max: {max(rss_changes):+.2f} MB")
    print(f"  - Số lần giảm: {sum(1 for x in rss_changes if x < 0)}/{len(rss_changes)}")
    
    print("\n" + "="*100)
    
    # Show dataset-level summary
    print("\nTÓM TẮT THEO DATASET:")
    print("="*100)
    
    for dataset in comp_df['Dataset'].unique():
        dataset_df = comp_df[comp_df['Dataset'] == dataset]
        print(f"\n{dataset}:")
        
        dataset_peak = [float(x.replace('+', '')) for x in dataset_df['Peak Δ (MB)']]
        dataset_rss = [float(x.replace('+', '')) for x in dataset_df['RSS Δ (MB)']]
        
        print(f"  Peak Memory: {sum(dataset_peak)/len(dataset_peak):+.2f} MB (avg)")
        print(f"  RSS Memory: {sum(dataset_rss)/len(dataset_rss):+.2f} MB (avg)")
        
        # Check if RAM decreased as expected
        if sum(dataset_peak)/len(dataset_peak) < 0:
            print(f"  ✅ Peak RAM giảm như kỳ vọng")
        else:
            print(f"  ❌ Peak RAM TĂNG (không như kỳ vọng)")
        
        if sum(dataset_rss)/len(dataset_rss) < 0:
            print(f"  ✅ RSS RAM giảm như kỳ vọng")
        else:
            print(f"  ⚠️  RSS RAM tăng (có thể do OS caching)")

if __name__ == '__main__':
    # Parse log file
    df = parse_memory_from_log('output.log')
    
    print(f"Đã parse {len(df)} dòng memory usage từ output.log")
    
    # Compare seed 1 vs seed 2
    if len(df[df['Seed'] == 1]) > 0 and len(df[df['Seed'] == 2]) > 0:
        compare_seeds(df, 1, 2)
    else:
        print("Không tìm thấy đủ dữ liệu cho seed 1 và seed 2")
        print(f"\nSeeds có trong data: {sorted(df['Seed'].unique())}")
