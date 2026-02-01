"""Analyze backtest CSV for Model A verification."""
import csv
from collections import Counter

def main():
    with open('backtest_5year_trades.csv', 'r') as f:
        reader = csv.DictReader(f)
        trades = list(reader)
    
    print('='*60)
    print('CSV Analysis - Model A Verification')
    print('='*60)
    print(f'\nTotal records: {len(trades)}')
    
    # Check for new columns
    columns = list(trades[0].keys())
    print(f'\nCSV Columns ({len(columns)} total):')
    for col in columns[:10]:
        print(f'  - {col}')
    print('  ...')
    
    # Row types (new field)
    if 'row_type' in columns:
        row_types = Counter(t['row_type'] for t in trades)
        print('\nRow Types:')
        for rt, count in row_types.most_common():
            print(f'  {rt}: {count}')
        
        # Count FINAL rows only for trade stats
        final_rows = [t for t in trades if t['row_type'] == 'FINAL']
        print(f'\n  Total unique trades (FINAL rows): {len(final_rows)}')
    
    # Count exit reasons
    reasons = Counter(t['exit_reason'] for t in trades)
    print('\nExit Reasons:')
    for reason, count in reasons.most_common():
        print(f'  {reason}: {count}')
    
    # Check lot sizes
    print('\nLot Size Analysis:')
    tp1_lots = [float(t['tp1_closed_lots']) for t in trades]
    tp2_lots = [float(t['tp2_closed_lots']) for t in trades]
    runner_lots = [float(t['runner_closed_lots']) for t in trades]
    
    tp1_with_lots = sum(1 for l in tp1_lots if l > 0)
    tp2_with_lots = sum(1 for l in tp2_lots if l > 0)
    runner_with_lots = sum(1 for l in runner_lots if l > 0)
    
    print(f'  TP1 closes with lots > 0: {tp1_with_lots}')
    print(f'  TP2 closes with lots > 0: {tp2_with_lots}')
    print(f'  Runner closes with lots > 0: {runner_with_lots}')
    
    # PnL verification (new fields)
    if 'pnl_event' in columns and 'pnl_total_trade' in columns:
        print('\nPnL Field Verification:')
        
        # Sum pnl_event for all rows
        total_event_pnl = sum(float(t['pnl_event']) for t in trades)
        
        # Sum pnl_total_trade for FINAL rows only
        final_rows = [t for t in trades if t['row_type'] == 'FINAL']
        total_trade_pnl = sum(float(t['pnl_total_trade']) for t in final_rows)
        
        print(f'  Sum of pnl_event (all rows): ${total_event_pnl:.2f}')
        print(f'  Sum of pnl_total_trade (FINAL only): ${total_trade_pnl:.2f}')
        
        if abs(total_event_pnl - total_trade_pnl) < 1.0:
            print('  [OK] PnL accounting is correct!')
        else:
            print('  [WARN] PnL mismatch - check accounting')
    
    # Sample trade lifecycle
    print('\n' + '-'*60)
    print('Sample Trade Lifecycle (first with TP1):')
    print('-'*60)
    sample_id = None
    for t in trades:
        if 'TP1' in t.get('exit_reason', '') or t.get('row_type') == 'TP1_PARTIAL':
            sample_id = t['trade_id']
            break
    
    if sample_id:
        for t in trades:
            if t['trade_id'] == sample_id:
                exit_time = t.get('exit_time', 'N/A')[:16] if t.get('exit_time') else 'N/A'
                row_type = t.get('row_type', 'N/A')[:12].ljust(12)
                reason = t.get('exit_reason', 'N/A')[:15].ljust(15)
                pnl_event = t.get('pnl_event', '0')
                pnl_total = t.get('pnl_total_trade', '0')
                print(f'{exit_time} | {row_type} | {reason} | event=${pnl_event} total=${pnl_total}')
    else:
        print('No TP1 exits found!')
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    if tp1_with_lots > 0 or tp2_with_lots > 0:
        print('[OK] Partial exits ARE working!')
        print(f'     {tp1_with_lots} TP1 partials, {tp2_with_lots} TP2 partials, {runner_with_lots} runner exits')
    else:
        print('[WARN] No partial exits with lots > 0 found')

if __name__ == '__main__':
    main()
