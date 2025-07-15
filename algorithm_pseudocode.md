# PACF-MI 动态协同特征选择算法伪代码

## 输入参数
- `time_series`: 输入时间序列数据 (1D array)
- `max_lag`: 最大滞后阶数 (int)
- `random_seed`: 随机种子 (int, 默认42)

## 输出
- `selected_features`: 筛选后的特征索引列表 (list)

## 算法流程

# 1. 数据预处理

def preprocess_data(series, max_lag):
    # 生成滞后特征矩阵
    features = []
    for lag in range(1, max_lag+1):
        features.append(series.shift(lag))  # 滞后特征
        
    # 添加时域/频域/时频域衍生特征 (根据应用场景)
    if industrial_scenario:
        features += extract_time_domain_features(series)   # 时域特征
        features += extract_freq_domain_features(series)  # 频域特征
        features += extract_time_freq_features(series)    # 时频域特征
    
    return pd.DataFrame(features).dropna()  # 转换为DataFrame并去除NaN
# 2. 数据特性检测
def detect_data_properties(X):
    # 平稳性检测 (ADF检验)
    adf_result = adfuller(X)
    is_nonstationary = adf_result[1] < 0.05  # p<0.05判为非平稳
    
    # 记忆性检测 (Hurst指数)
    hurst = compute_hurst(X)  # 使用R/S分析法
    has_strong_memory = hurst > 0.7
    
    return is_nonstationary, has_strong_memory
# 3. 动态权重调整
def determine_alpha(is_nonstationary, has_strong_memory):
    # 优先级: 非平稳性 > 强记忆性
    if is_nonstationary:
        alpha_range = [0.2, 0.4]  # 非平稳数据范围
    elif has_strong_memory:
        alpha_range = [0.6, 0.8]  # 强记忆数据范围
    else:
        alpha_range = [0.4, 0.6]  # 默认范围
        
    # 网格搜索确定最优alpha
    best_alpha = grid_search_alpha(alpha_range)
    return best_alpha

def grid_search_alpha(alpha_range):
    best_score = -np.inf
    best_alpha = 0.5
    
    for alpha in np.arange(alpha_range[0], alpha_range[1]+0.1, 0.1):
        # 1. 计算特征评分
        scores = compute_feature_scores(features, alpha)
        
        # 2. 特征选择
        selected = select_features(scores)
        
        # 3. 训练模型并评估
        model = train_model(features[selected], target)
        score = evaluate_model(model, test_set)
        
        # 4. 更新最优值
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    return best_alpha
   #  4. 混合特征评分
   def compute_feature_scores(features, alpha):
    scores = []
    
    # 计算PACF
    pacf_values = pacf(features, nlags=max_lag)
    
    # 计算互信息MI
    mi_values = []
    for feature in features.columns:
        mi = mutual_info_regression(features[feature].values.reshape(-1,1), 
                                   target)
        mi_values.append(mi)
    
    # MI归一化
    mi_max = max(mi_values)
    mi_norm = [mi / mi_max for mi in mi_values]
    
    # 计算混合评分
    for i in range(len(features.columns)):
        score = alpha * abs(pacf_values[i]) + (1-alpha) * mi_norm[i]
        scores.append(score)
    
    return scores
   # 5. 特征选择
   def select_features(scores, threshold=0.65):
    # 动态阈值 (前20%分位数)
    if threshold == 'auto':
        threshold = np.percentile(scores, 80)
    
    # 前向选择 (按评分降序)
    candidate_features = [i for i in range(len(scores))]
    candidate_features.sort(key=lambda i: scores[i], reverse=True)
    
    # 初始筛选
    selected = [idx for idx in candidate_features if scores[idx] >= threshold]
    
    # 后向剪枝 (相关系数阈值0.8)
    pruned_features = []
    for i in selected:
        keep = True
        for j in pruned_features:
            if abs(pearsonr(features[i], features[j])[0]) > 0.8:
                keep = False
                break
        if keep:
            pruned_features.append(i)
    
    return pruned_features
   # 6. 主算法
   def pacf_mi_feature_selection(time_series, max_lag=20, random_seed=42):
    # 设置随机种子确保可复现
    np.random.seed(random_seed)
    
    # Step 1: 数据预处理
    feature_df = preprocess_data(time_series, max_lag)
    
    # Step 2: 数据特性检测
    is_nonstationary, has_strong_memory = detect_data_properties(feature_df)
    
    # Step 3: 确定动态权重alpha
    alpha = determine_alpha(is_nonstationary, has_strong_memory)
    
    # Step 4: 计算混合评分
    scores = compute_feature_scores(feature_df, alpha)
    
    # Step 5: 特征选择
    selected_features = select_features(scores)
    
    return selected_features
