import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_flavor as pf
import scipy.stats as stats
import statsmodels.stats.api as sms

def bootstrap(A, B):
    combined = A + B
    resampled = np.random.choice(combined, size=len(combined), replace=True)
    resampled_A = resampled[:len(A)]
    resampled_B = resampled[len(A):]
    return resampled_A, resampled_B

def cohen_d(A, B):
    n1, n2 = len(A), len(B)
    std1, std2 = np.std(A, ddof=1), np.std(B, ddof=1)
    mean1, mean2 = np.mean(A), np.mean(B)
    numerator = mean1-mean2
    pooled_sd = np.sqrt(((n1-1)*(std1**2) + (n2-1)*(std2**2)) / (n1+n2-2))
    d = numerator / pooled_sd
    return d

def combT(a,b):
    universal_set = sorted(a + b)
    combinations = set(itertools.combinations(universal_set, len(a)))
    groupings = []
    for combination in combinations:
        temp_list = universal_set.copy()
        for element in combination:
            temp_list.remove(element)
        groupings.append((list(combination), temp_list))
    return groupings

from IPython.display import display_html
def display_side_by_side(*args, names=None):
    html_str=''
    html_str+='<table>'
    for i, df in enumerate(args):
        html_str+='<td>'
        if names:
            name_str = names[i]+'<br/>'
            html_str+=name_str
        html_str+=df.to_html()
        html_str+='</td>'
    html_str+='</table></body>'
    display_html(html_str.replace('table','table style="display:inline" cellpadding=100'),raw=True)

def ecdf(data, group_by=None, targets=None, ax=None):
    """Produces ECDF graphs for input data. Inputs can be 1d array-like, pandas Series, or
    pandas DataFrame. If a DataFrame is passed, group_by and targets may be set for group 
    comparisons. If no target is set for a DataFrame, all columns will be graphed."""
    if group_by is not None:
        if type(data) == pd.core.frame.DataFrame:
            print("Grouping DataFrame by {}".format(group_by))
            print("Target Features:", targets)
            if type(targets) == str:
                targets = [targets]
            else:
                try:
                    it = iter(targets)
                except:
                    targets = [targets]
            cols = targets + [group_by]
            data = data[cols]
            variables = data.columns[:-1]
            data = data.groupby(group_by)
        else:
            return("Error: only DataFrame input works with group_by functionality")
    else:      
        if type(data) == pd.core.series.Series:
            variables = [data.name]
        elif type(data) == pd.core.frame.DataFrame:
            if targets is None:
                variables = list(data.columns)
            else:
                if type(targets) == str:
                    targets = [targets]
                else:    
                    try:
                        it = iter(targets)
                    except:
                        targets = [targets]
                print("Target Features:", targets)
                variables = targets
        elif type(data) == pd.core.groupby.generic.DataFrameGroupBy:
            variables = list(data.obj.columns)
        else:
            data = pd.Series(data, name='data')
            variables = [data.name]
    
    
    if type(data) == pd.core.groupby.generic.DataFrameGroupBy:
        for variable in variables:
            if not ax:
                fig, ax = plt.subplots(figsize=(12,8))
            max_x = 0
            for name, group in data:
                x = np.sort(group[variable])
                n = len(group)
                y = np.arange(1, n+1) / n
                ax.plot(x, y, marker='.', label=name, alpha=0.6)
                if max(x) > max_x:
                    max_x = max(x)
                    #max_x = 0
            ax.axhline(y=0.5, ls=':', color='gray')
            ax.axhline(y=0.05, ls=':', color='gray')
            ax.axhline(y=0.95, ls=':', color='gray')
            ax.annotate('0.5', xy=(max_x, 0.47))
            ax.annotate('0.95', xy=(max_x, 0.92))
            ax.annotate('0.05', xy=(max_x, 0.02))
            ax.legend()
            plt.title("ECDF for feature: {}".format(variable), color='gray')
            plt.show()
                
    else:
        n = len(data)
        y = np.arange(1, n+1) / n
        if not ax:
            fig, ax = plt.subplots(figsize=(12,8))
        max_x = 0
        for variable in variables:
            if type(data) == pd.core.series.Series:
                x = np.sort(data)
                string = variable
            else:
                x = np.sort(data[variable])
                string = 'Data'
            ax.plot(x, y, marker='.', label=variable)
            if max(x) > max_x:
                max_x = max(x)
        ax.axhline(y=0.5, ls=':', color='gray')
        ax.axhline(y=0.05, ls=':', color='gray')
        ax.axhline(y=0.95, ls=':', color='gray')
        ax.annotate('0.5', xy=(max_x, 0.47))
        ax.annotate('0.95', xy=(max_x, 0.92))
        ax.annotate('0.05', xy=(max_x, 0.02))
        plt.title("ECDF for {}".format(string), color='gray')
        plt.legend()
        plt.show()

def f_test(var1, var2, df1, df2, alternate='both'):
    F = var1/var2
                
    if alternate == 'both':
        if F > 1:
            p = stats.f.sf(F, df1, df2) * 2
        elif F <= 1:
            p = stats.f.cdf(F, df1, df2) * 2
    elif alternate == 'lower':
        p = stats.f.cdf(F, df1, df2)
    elif alternate == 'higher':
        p = stats.f.sf(F, df1, df2)
    else:
        return ("Error: invalid alternate hypothesis. Choices: 'both', 'lower', 'higher'")
    
    return p

def f_test_groups(data, group_var, target, alternate='both'):
    groups = data.groupby(group_var)[target]
    scores = {}
    
    for name1, group1 in groups:
        group1_scores = {}
        var1 = np.var(group1, ddof=1)
        df1 = len(group1) - 1
        
        for name2, group2 in groups:
            
            if name2 != name1:
                var2 = np.var(group2, ddof=1)
                df2 = len(group2) - 1
                p = f_test(var1, var2, df1, df2)
                group1_scores[name2] = p
                
        scores[name1] = group1_scores
    
    scores = pd.DataFrame(scores).sort_index()
    
    return scores         

def goldfeld_quandt(dataframe, target, model, ax):
    temp = dataframe.sort_values(by=target).reset_index(drop=True)
    lwr_thresh = temp[target].quantile(q=.45)
    upr_thresh = temp[target].quantile(q=.55)
    middle_10percent_indices = temp[(temp[target] >= lwr_thresh) & (temp[target]<=upr_thresh)].index
    indices = [x-1 for x in temp.index if x not in middle_10percent_indices]
    if not ax:
        fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(temp[target].iloc[indices], model.resid.iloc[indices])
    ax.set_xlabel(target)
    ax.set_ylabel('Model Residuals')
    ax.set_title("Residuals versus {}".format(target), color='gray')
    ax.axvline(x=lwr_thresh, ls=':',linewidth=2, color='gray')
    ax.axvline(x=upr_thresh, ls=':',linewidth=2, color='gray')
    if not ax:
        plt.show()
    test = sms.het_goldfeldquandt(model.resid.iloc[indices], model.model.exog[indices])
    results = pd.DataFrame(index=['Goldfeld-Quandt'], columns=['F_statistic', 'p_value'])
    results.loc['Goldfeld-Quandt','F_statistic'] = test[0]
    results.loc['Goldfeld-Quandt','p_value'] = test[1]
    return results        

# A function for metropolis MCMC algorithm:
def metropolis(data1, theta_seed1, theta_std1, data2=None, theta_seed2=None, theta_std2=None, samples=10000):
    theta_curr1 = theta_seed1
    posterior_thetas1 = []
    graph_thetas1 = []
    scaleA = np.std(data1, ddof=1)
    n1 = len(data1)
    calc_mean1 = np.mean(data1)
    post_std = theta_std1
    
    if data2 is not None:
        theta_curr2 = theta_seed2
        calc_mean2 = np.mean(data2)
        posterior_thetas2 = []
        theta_diffs = []
        effect_sizes = []
        graph_thetas2 = []
        scaleB = np.std(data2, ddof=1)
        actual_diff = calc_mean1 - calc_mean2
        actual_effect = actual_diff/np.sqrt((scaleA**2 + scaleB**2)/2)
        print("Performing MCMC for two groups")
        print("Mean of Group 1:", calc_mean1)
        print("Mean of Group 2:", calc_mean2)
        print("Measured Mean Difference:", actual_diff)
        print("Measured Effect Size:", actual_effect)
    
    for i in range(samples):
        theta_prop1 = np.random.normal(loc=theta_curr1, scale=post_std)
        likelihood_prop1 = 1
        if i == 0:
            likelihood_curr1 = 1
        #scaleA = min([np.random.normal(loc=scaleA, scale=0.05), 0])
        if data2 is not None:
            theta_prop2 = np.random.normal(loc=theta_curr2, scale=theta_std2)
            likelihood_prop2 = 1
            likelihood_curr2 = 1
            #scaleB = min([np.random.normal(loc=scaleB, scale=0.05), 0])
        #print(theta_prop1)
        
        #data1 = np.random.normal(loc=calc_mean1, scale=scaleA, size=n1)
        #mean1 = data1.mean()
        for datum in data1:
            pd_prop = stats.norm.pdf(x=datum, loc=theta_prop1, scale=scaleA)
            likelihood_prop1 *= pd_prop
            if i == 0:
                pd_curr = stats.norm.pdf(x=datum, loc=theta_curr1, scale=scaleA)
                likelihood_curr1 *= pd_curr
        
        posterior_prop1 = likelihood_prop1 * stats.norm.pdf(x=theta_prop1, loc=theta_curr1, scale=theta_std1)
        if i == 0:
            posterior_curr1 = likelihood_curr1 * stats.norm.pdf(x=theta_curr1, loc=theta_curr1, scale=theta_std1)
        #posterior_prop1 = likelihood_prop1 * stats.uniform.pdf(x=theta_prop1, loc=theta_curr1, scale=theta_std1)
        #posterior_curr1 = likelihood_curr1 * stats.uniform.pdf(x=theta_curr1, loc=theta_curr1, scale=theta_std1)
        
        if data2 is not None:
            for datum in data2:
                pd_prop = stats.norm.pdf(x=datum, loc=theta_prop2, scale=scaleB)
                likelihood_prop2 *= pd_prop
                if i == 0 :
                    pd_curr = stats.norm.pdf(x=datum, loc=theta_curr2, scale=scaleB)
                    likelihood_curr2 *= pd_curr
                
            posterior_prop2 = likelihood_prop2 * stats.norm.pdf(x=theta_prop2, loc=theta_curr2, scale=theta_std2)
            if i == 0:
                posterior_curr2 = likelihood_curr2 * stats.norm.pdf(x=theta_curr2, loc=theta_curr2, scale=theta_std2)
            #posterior_prop2 = likelihood_prop2 * stats.uniform.pdf(x=theta_prop2, loc=theta_curr2, scale=theta_std2)
            #posterior_curr2 = likelihood_curr2 * stats.uniform.pdf(x=theta_curr2, loc=theta_curr2, scale=theta_std2)
        
        # Prevents division by zero:
        if posterior_curr1 == 0.0:
            posterior_curr1 = 2.2250738585072014e-308
        if data2 is not None and posterior_curr2 == 0.0:
            posterior_curr2 = 2.2250738585072014e-308
            
        p_accept_theta_prop1 = posterior_prop1/posterior_curr1
        rand_unif = np.random.uniform()
        if p_accept_theta_prop1 >= rand_unif:
            #post_mean, post_std, posterior = make_posterior(calc_mean1, theta_prop1, scaleA, post_std)
            theta_curr1 = theta_prop1
            posterior_curr1 = posterior_prop1
            #scaleA = scaleA
        posterior_thetas1.append(theta_curr1)
        if i % (samples/10) == 0:
            graph_thetas1.append(theta_curr1)
        
        if data2 is not None:
            #print(posterior_prop2, posterior_curr2)
            p_accept_theta_prop2 = posterior_prop2/posterior_curr2
            rand_unif = np.random.uniform()
            if p_accept_theta_prop2 >= rand_unif:
                theta_curr2 = theta_prop2
                posterior_curr2 = posterior_prop2
                
            posterior_thetas2.append(theta_curr2)
            theta_diff = theta_curr1 - theta_curr2
            theta_diffs.append(theta_diff)
            effect_sizes.append(theta_diff/np.sqrt((scaleA**2 + scaleB**2)/2))
        
            if i % (samples/10) == 0:
                graph_thetas2.append(theta_curr2)
                
    if data2 is not None:
        # Visualizing results of MCMC
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, 
                                                                 ncols=2,
                                                                 figsize=(12,12))
        ax1.hist(data1, density=True, alpha=0.6)
        ax1.set_title("Data Group 1 w. Post. Pred")
        ax1.axvline(x=calc_mean1, ls=':', label='Group 1 Mean: {}'.format(calc_mean1))
        ax1.legend()
        xs = np.linspace(data1.min(), data1.max(), 1000)
        for theta in graph_thetas1:
            ys = stats.norm.pdf(xs, loc=theta, scale=scaleA)
            ax1.plot(xs, ys, color='gray')
        ax2.hist(posterior_thetas1, density=True, alpha=0.6)
        ax2.set_title("Posterior for Theta, Group 1")
        ax2.axvline(x=np.mean(posterior_thetas1), ls=':', label='Mean of Posterior 1: {}'.format(np.mean(posterior_thetas1)))
        ax2.legend()
        ax3.hist(data2, density=True, alpha=0.6)
        ax3.set_title("Data Group 2 w. Post. Pred")
        ax3.axvline(x=calc_mean2, ls=':', label='Group 2 Mean: {}'.format(calc_mean2))
        ax3.legend()
        xs = np.linspace(data2.min(), data2.max(), 1000)
        for theta in graph_thetas2:
            ys = stats.norm.pdf(xs, loc=theta, scale=scaleB)
            ax3.plot(xs, ys, color='gray')
        ax4.hist(posterior_thetas2, density=True, alpha=0.6)
        ax4.set_title("Posterior for Theta, Group 2")
        ax4.axvline(x=np.mean(posterior_thetas2), ls=':', label='Mean of Posterior 2:: {}'.format(np.mean(posterior_thetas2)))
        ax4.legend()
        ax5.hist(theta_diffs, density=True, alpha=0.6)
        ax5.set_title("Differences btw Theta 1 and 2")
        ax5.axvline(x=np.mean(theta_diffs), ls=':', label='Mean Difference: {}'.format(np.mean(theta_diffs)))
        ax5.legend()
        ax6.hist(effect_sizes, density=True, alpha=0.6)
        ax6.set_title("Effect Sizes")
        ax6.axvline(x=np.mean(effect_sizes), ls=':', label='Mean Effect Size: {}'.format(np.mean(effect_sizes)))
        ax6.legend()
        plt.show()
        
        # Producing probability of null hypothesis:
        sizes = np.array(theta_diffs)
        sizes_mu = sizes.mean()
        sizes_std = sizes.std()
        conf_interval = stats.norm.interval(0.95, loc=sizes_mu, scale=sizes_std)
        if np.mean(theta_diffs) >= 0:
            calc_p_val = ((sum(sizes < 0) / len(sizes)) * 2)
            norm_p_val = (stats.norm.cdf(0, loc=sizes_mu, scale=sizes_std) * 2)
        else:
            calc_p_val = ((sum(sizes > 0) / len(sizes)) * 2)
            norm_p_val = (stats.norm.sf(0, loc=sizes_mu, scale=sizes_std) * 2)

        print("P_value numerically:", calc_p_val)
        print("P_value from normal dist:", norm_p_val)
        print("95% Confidence Interval for Mean Difference:", conf_interval)
        
        return theta_diffs
        
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        ax1.set_title('Group 1 Data w Post. Pred')
        ax1.hist(data1, density=True, alpha=0.6)
        ax1.axvline(x=calc_mean1, ls=':', color='g', label='Measured mean: {}'.format(calc_mean1))
        xs = np.linspace(min(data1), max(data1), 1000)
        for theta in graph_thetas1:
            ys = stats.norm.pdf(xs, loc=theta, scale=scaleA)
            ax1.plot(xs, ys, color='gray')
        ax1.legend()
        
        ax2.set_title('MCMC Mean Frequencies')
        ax2.hist(posterior_thetas1, density=True, alpha=0.6)
        mcmc_theta1 = np.mean(posterior_thetas1)
        ax2.axvline(x=mcmc_theta1, ls=':', color='g', label='MCMC mean: {}'.format(mcmc_theta1))
        ax2.legend()
        plt.show()
        
        return posterior_thetas1

def norm_pdf(x, mu, std):
    var = std**2
    part1 = 1/(np.sqrt(2*np.pi)*std)
    part2 = np.exp(-1*((x-mu)**2)/(2*var))
    pd = part1 * part2
    return pd

# A function to run permutation tests:
def permutation(dataframe, feature, target, control=0.0, alternate='both'):
    
    for name, group in dataframe.groupby(feature)[target]:
        if len(group) == 0:
            continue
        p_vals = {}
        
        if name == control:
            control_group = group
            # To manage exploding numbers of combinations, need to take sample if n too large
            N = len(control_group)
            if N > 50:
                # Use Slovin's formula to figure out the sample size that we will need
                e = .05
                n = int(round((N / (1 + N*(e**2))), 0))
                print("Sampling control group with size {}".format(n))
                control_group = np.random.choice(control_group, size=n, replace=False)
        else:
            mean_diff = group.mean()
            further_diffs = 0
            group_dict = {}
            groupings = combT(list(group), list(control_group))
            print("Number of Groupings for {} group:".format(name), len(groupings))
            for grouping in groupings:
                mean1 = np.mean(grouping[0])
                mean2 = np.mean(grouping[1])
                diff = mean1 - mean2
                if alternate == 'lower':
                    if diff <= mean_diff:
                        further_diffs += 1
                elif alternate == 'both':
                    if np.abs(diff) >= np.abs(mean_diff):
                        further_diffs += 1
                elif alternate == 'higher':
                    if diff >= mean_diff:
                        further_diffs += 1
                else:
                    print("Error: invalid alternate hypothesis. Options are 'both', 'lower', 'higher'")
            p_val = further_diffs / len(groupings)
            group_dict['p-value'] = p_val
            p_vals[name] = group_dict
            
    test_results = pd.DataFrame.from_dict(p_vals)
    return test_results

def pooled_variance(groups):
    info = {}
    names = []
    for name, group in groups:
        names.append(name)
        info[name] = {}
        info[name]['n'] = len(group)
        info[name]['var'] = group.var(ddof=1)
    k = len(info.keys())
    numer = sum([(info[name]['n'] - 1)*info[name]['var'] for name in names])
    denom = sum([info[name]['n'] for name in names]) - k
    pooled_var = np.sqrt(numer/denom)
    return pooled_var

def dunnets_tstat(expr, ctrl, pooled_var):
    expr_mean = expr.mean()
    ctrl_mean = ctrl.mean()
    n1, n2 = len(expr), len(ctrl)
    return (expr_mean - ctrl_mean) / np.sqrt(pooled_var * ((1/(n1))+(1/(n2))))