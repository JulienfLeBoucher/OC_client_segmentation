import pandas as pd

def make_summary(datasets):
    summary = pd.DataFrame({},)
    summary['dataset_name'] = datasets.keys()
    summary['columns_names'] = [list(dataset.columns)
                                for dataset in datasets.values()]
    summary['rows_num'] = [dataset.shape[0] for dataset in datasets.values()]
    summary['cols_num'] = [dataset.shape[1] for dataset in datasets.values()]
    summary['total_duplicates'] = [dataset.duplicated().sum() 
                                   for dataset in datasets.values()]
    summary = summary.set_index(['dataset_name'])
    return summary


def display_summary(datasets):
    summary = make_summary(datasets)
    display(summary.style.background_gradient(cmap='Purples'))
    return None


### MERGING DFS
def removekey(d, key):
    """ Copy and return the dictionary d with the deleted element
    at key ."""
    r = dict(d)
    del r[key]
    return r


# Had to use the following because locals() seemed not to work with df?
def arguments():
        """Returns a tuple containing :
           - a dictionary of the calling function's
           named arguments, and ;
           - a list of calling function's unnamed
           positional arguments.
        """
        from inspect import getargvalues, stack
        posname, kwname, args = getargvalues(stack()[1][0])[-3:]
        posargs = args.pop(posname, [])
        args.update(args.pop(kwname, []))
        return args, posargs
    

def merge_and_display(left, right, on=None, how='left', validate=None):
    """ Merge from left and display nulls values and shapes 
    in order to understand what occurs during the merging process."""
    args, _ = arguments()
    df = pd.merge(**args)
    print(f"shape before merging : {left.shape}")
    print(f"shape of the right df: {right.shape}")
    print(f"shape after merging : {df.shape}")
    display(df.notnull().mean())
    return df

#### Checking prices consistency
def order_total_price(grp):
    """ search the price and the freight value of each item and sum it.
    
    remark : the min aggregate function, could be max or mean,
    it wont change the result."""
    return round(grp.groupby('order_item_id')[['price', 'freight_value']]
            .min().sum().sum(), 2)
    
    
def total_payment_value(grp):
    """ search the values for each sequential payment and sum it
    
    remark : the min aggregate function, could be max or mean,
    it wont change the result."""
    return round(grp.groupby('payment_sequential')['payment_value']
            .min().sum(), 2)
    
    
def is_total_price_equal_to_payment(grp):
    """ Check wether the payment and the total order price match.
    A 5 centavos difference is tolerated."""
    return (abs(total_payment_value(grp) - order_total_price(grp)) <= 0.05) 

def price_payment_diff(grp):
    """ Test if the order total price is not zero and if price and payment
    are different for more than 5 centavos."""
    return ((order_total_price(grp) != 0) 
            & ~is_total_price_equal_to_payment(grp))
    
def display_client(id, df):
    display(df.query('customer_unique_id == @id'))
    return None

### Features engineering
def client_orders_summary(client_info):
    """ Return a df with one line per order made by the client"""
    return (client_info
            .groupby('order_id')
            .agg(
                order_status=('order_status', 'first'),
                order_purchase_timestamp=('order_purchase_timestamp', 'first'),
                total_order_cost=('total_order_cost', 'first'),
                cost_minus_payment=('cost_minus_payment', 'first'),
                number_of_items=('order_item_id', 'max'),
                review_score_min=('review_score', 'min'),
                review_score_mean=('review_score', 'mean'),
                review_score_max=('review_score', 'max'),
                payment_installments=('payment_installments', 'max'),
            )
    )
    
def relatives_to_now(orders_summary, now):
    """ Return the t-uple 
    (
        'number_of_purchases_last_365_days',
        'number_of_purchases_last_90_days',
        'number_of_purchases_last_30_days',
        'elapsed_days_since_last_purchase'
    ) 
    """
    orders_summary['now'] = now
    orders_summary['days_since_order'] = ((orders_summary.now
                                           - (orders_summary
                                              .order_purchase_timestamp))
                                          .dt.days)
    
    return (
                len(orders_summary.query("days_since_order <= 365")),
                len(orders_summary.query("days_since_order <= 90")),
                len(orders_summary.query("days_since_order <= 30")),
                orders_summary['days_since_order'].min(),
    )