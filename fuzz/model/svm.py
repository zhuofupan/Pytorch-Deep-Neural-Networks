# -*- coding: utf-8 -*-

# SVC 与 SVR 共有：
'''
    >>> Parameters(**kwargs) >>>
    C:    float, optional (default=1.0) - L2正则化系数
    kernel:    string, optional (default=’rbf’) - ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    degree:    int, optional (default=3) - ‘poly’核的参数
    gamma:    {‘scale’, ‘auto’} or float, optional (default=’scale’) - ‘rbf’, ‘poly’, ‘sigmoid’核的参数
            ‘scale’: gamma = 1 / (n_features * X.var())
            ‘auto’: gamma = 1 / n_features
    coef0:    float, optional (default=0.0) - ‘poly’, ‘sigmoid’核的参数
    shrinking:    boolean, optional (default=True) - 是否使用缩小的启发式方法
    tol:    float, optional (default=1e-3) - 停止搜索条件
    cache_size:    float, optional - 指定内核缓存的大小（以MB为单位）
    verbose:    bool, default: False - 启用详细输出
    max_iter:    int, optional (default=-1) - 对求解器内的迭代进行硬性限制，或者为-1（无限制）
    ----------------------------------------------------------------------------------------------------
    >>> Attributes(self.) >>>
    support_:    (n_SV) - 支持向量索引
    support_vectors_:    (n_SV, n_features) - 支持向量
    dual_coef_:    [n_class-1, n_SV] - 决策函数中支持向量的系数
    coef_:    [n_class * (n_class-1) / 2, n_features] - 分配给特征的权重
    fit_status_:    int - 正确fit时为1
    intercept_:    (n_class * (n_class-1) / 2,) - 决策函数中的常数
    ----------------------------------------------------------------------------------------------------
    >>> Methods(self.) >>>
    fit(self, X, y[, sample_weight]) - 训练函数
    get_params(self[, deep]) - 获取SVM的设置参数
    predict(self, X) - 测试
    score(self, X, y[, sample_weight]) - 平均正确率 / R2
    set_params(self, \*\*params) - 设置参数
'''
# SVC 独有：
'''    
    >>> Parameters(**kwargs) >>>
    class_weight:    {dict, ‘balanced’}, optional - 对于SVC，将类i的参数C设置为 class_weight [i] * C。 
            如果未给出，则所有类都应具有权重一。 ‘balanced’模式使用y的值自动将权重与输入数据中的类频率调整为
            n_samples /(n_classes * np.bincount(y))
    probability:    boolean, optional (default=False) - 是否启用概率估计。 必须在调用fit之前启用此功能，
            因为该方法内部使用5倍交叉验证，因此会减慢该方法的速度，并且predict_proba可能与预测不一致
    decision_function_shape:    ‘ovo’, ‘ovr’, default=’ovr’ - 决策函数形状
            ‘ovo’: 一对一 (n_samples, n_classes * (n_classes - 1) / 2)
            ‘ovr’: 一对多 (n_samples, n_classes)
    break_ties:    bool, optional (default=False) - 如果为true，decision_function_shape ='ovr'，
            并且类数> 2，则预测将根据Decision_function的置信度值 break ties； 否则，将返回绑定类中的第一类
    random_state:    int, RandomState instance or None, optional (default=None) - 伪随机数生成器的种子
    ----------------------------------------------------------------------------------------------------
    >>> Attributes(self.) >>>
    n_support_:    [n_class] - 每个类别的支持向量个数
    classes_:    (n_classes,) - 类别标签
    class_weight_:    (n_class,) - 每个类的参数C的乘数。 根据class_weight参数进行计算
    shape_fit_:    (n_dimensions_of_X,) - 训练向量X的数组尺寸
    ----------------------------------------------------------------------------------------------------
    >>> Methods(self.) >>>
    decision_function(self, X) - 评估X的决策函数
'''
# SVR 独有：
'''
    >>> Parameters(**kwargs) >>>
    epsilon:    float, optional (default=0.1) - epsilon-SVR模型中的Epsilon
'''
from sklearn.svm import SVC, SVR

class SVM():  
    def __init__(self, task = 'cls', **kwargs):
        if task == 'cls':
            self.svm = SVC(**kwargs)
            self._name = 'SVC'
        elif task == 'prd':
            self.svm = SVR(**kwargs)
            self._name = 'SVR'
            
    def decision_function(self, X):
        '''
            X (n_samples, n_features)
            return:  X (n_samples, n_classes * (n_classes-1) / 2)
        '''
        if self._name == 'SVC':
            return self.svm.decision_function(X)
    
    def fit(self, X, y, sample_weight = None):
        '''
            X (n_samples, n_features)
            y (n_samples,)
            sample_weight (n_samples,)
        '''
        return self.svm.fit(X,y,sample_weight)
    
    def get_params(self, deep=True):
        return self.svm.get_params(deep)
    
    def predict(self, X):
        return self.svm.predict(X)
    
    def score(self, X, y, sample_weight=None):
        '''
            X (n_samples, n_features)
            y (n_samples,) or (n_samples, n_outputs)
            sample_weight (n_samples,), default=None
        '''
        return self.svm.score(X, y, sample_weight)
        
    def set_params(self, **params):
        '''
            **params dict
        '''
        return self.svm.set_params(**params)