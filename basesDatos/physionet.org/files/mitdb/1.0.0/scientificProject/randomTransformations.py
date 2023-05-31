def randomTransformation(random_number,batch_x,batch_y):
    if random_number == 0:
        print("no hago transformacion")

    elif random_number == 1:
        print("pasobajo")

    elif random_number == 2:
        print("pasoalto")

    elif random_number == 3:
        print("bandpass")

    elif random_number == 4:
        print("baseline wander")
    elif random_number == 5:
        print("ruido 50 hz")

    elif random_number == 6:
        print("añadir ruido con un srn aleatorio")

    elif random_number == 7:
        print("desplazamiento aleatorio")

    return batch_x,batch_y