def gradient_descent(f, gradient, x, j_real, eps=1e-6, max_iter=100, initial_alpha=0.1):
    """
    Aquesta funci� implementa l'algorisme de descens pel gradient.
    
    :param f: Funci� a minimitzar
    :param x: Punt inicial
    :param eps: Moviment m�nim realitzat abans de parar
    :param max_iter: Iteracions m�ximes a realitzar
    :param initial_alpha: Pas inicial a cada iteraci�, corresponent al punt 3 anterior
    :param verbose: En case de ser True, la funci� ha d'imprimir el nombre d'iteracions fetes
        abans de retornar
    :return: La funci� retornar� el punt m�nim.

    """
    h = np.zeros(1)
    point = x #Point ser� l'�ltim punt trobat i next_point el seg�ent que calculem a partir de point
    points = np.array([point]) #Definim un vector amb els punts trobats, per comen�ar el punt inicial
    next_point = x + eps #Per poder entrar al bucle
    error = [mean_error(h, point, h, j_real)]
    alpha = initial_alpha
    change_alpha=[0]
    for iters in tqdm_notebook(range(max_iter)):
        point = points[-1]
        grad_p = gradient(point)
        next_point = point-alpha*grad_p
        #next_point = interval(next_point)
        while f(next_point) > f(point):
            alpha /= 1.5
            next_point = point-alpha*grad_p
            #next_point = interval(next_point)
            change_alpha.append(iters)
        points = np.append(points, [next_point], axis=0)
        error.append(mean_error(h, next_point, h, j_real))
        #if iters%10 == 0:
            #print(next_point, f(next_point), error[-1])
        #if np.linalg.norm(point-next_point)>eps:
            #break
        
    return points[-1], f(points[-1]), np.array(error), np.array(change_alpha)