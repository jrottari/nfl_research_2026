import warnings; warnings.filterwarnings('ignore')
import nflreadpy as nfl

df = nfl.load_ff_rankings(type='all')
for ecr in ['rp', 'ro', 'bp', 'wo']:
    sub = df.filter(df['ecr_type'] == ecr)
    print(f'ecr_type={ecr}: page_types={sub["page_type"].unique().to_list()[:5]}')
    print(f'  scrape_dates sample: {sub["scrape_date"].unique().to_list()[:5]}')
    sub2 = sub.select(['scrape_date','player','pos','ecr','page_type'])
    print(sub2.head(2))
    print()
