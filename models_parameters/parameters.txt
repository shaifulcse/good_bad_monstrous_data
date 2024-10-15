LogisticRegression(random_state=0, max_iter = 10000, verbose = True)

RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5,
                                        min_samples_leaf=5, max_features = 8)
DecisionTreeClassifier(random_state=0, max_depth=7)

AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)

MLPClassifier(max_iter= 1000, solver='adam', alpha=0, learning_rate_init = 0.0001,
                            hidden_layer_sizes=(10), warm_start= False, verbose = True)



