from game_controller_BOUNCE import game_controller

''' Enter stuff here to customize the experiment... '''
settings = {
            "game_size" : [22,10],
            "n_players" : 4,
            "run-id" : "my_first_experiment",
            }

controller = game_controller(settings=settings)
controller.train(n_steps=1000)
