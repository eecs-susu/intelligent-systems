class InternalCombustionEngine(object):
    def __init__(self, cylinders):
        possible_cylinders = [4, 6, 8]
        if cylinders not in possible_cylinders:
            raise ValueError('Cylinders number mast be in range {}'.format(possible_cylinders))
        self._cylinders = []


class Ignition(object):
    def __init__(self):
        self._subscribers = set()

    def subscribe(self, callback):
        self._subscribers.add(callback)

    def unsubscribe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _generate_event(self, event_text):
        for callback in self._subscribers:
            callback(event_text)

    def ignite(self):
        self._generate_event('Подача сигнала для старта работы двигателя')


class ExhaustSystem(object):
    def __init__(self):
        self._subscribers = set()

    def subscribe(self, callback):
        self._subscribers.add(callback)

    def unsubscribe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _generate_event(self, event_text):
        for callback in self._subscribers:
            callback(event_text)

    def exhaust(self):
        self._generate_event('Вывод газов в атмосферу')


class FuelSystem(object):
    def __init__(self):
        self._subscribers = set()

    def subscribe(self, callback):
        self._subscribers.add(callback)

    def unsubscribe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _generate_event(self, event_text):
        for callback in self._subscribers:
            callback(event_text)

    def provide_fuel(self):
        self._generate_event('Подача топлива')


class FourStrokeEngine(InternalCombustionEngine):
    def __init__(self, cylinders):
        super(FourStrokeEngine, self).__init__(cylinders)
        self._cylinders = []
        self._init_cylinders(cylinders)
        self._ignition = Ignition()
        self._init_ignition()
        self._exhaust_system = ExhaustSystem()
        self._init_exhaust_system()
        self._fuel_system = FuelSystem()
        self._init_fuel_system()
        self._started = False

    def _log(self, name):
        def log(text, **kwargs):
            print('{}: {}'.format(name, text))
            if 'exhaust' in kwargs:
                self._exhaust_system.exhaust()

            if 'fuel' in kwargs:
                self._fuel_system.provide_fuel()

        return log

    def _init_cylinders(self, cylinders):
        for i in range(cylinders):
            self._cylinders.append(FourStrokeCylinder(bool(i % 2)))
            self._cylinders[i].subscribe(self._log('Цилиндр {}'.format(i + 1)))

    def _init_ignition(self):
        self._ignition.subscribe(self._log('Система зажигания'))

    def _init_fuel_system(self):
        self._fuel_system.subscribe(self._log('Система подачи топлива'))

    def _init_exhaust_system(self):
        self._exhaust_system.subscribe(self._log('Выхлопная система'))

    def process_stroke(self):
        if not self._started:
            self._ignition.ignite()
            self._started = True

        process_func = map(lambda e: e.process_stroke(), self._cylinders)
        list(process_func)


class Cylinder(object):
    def process_stroke(self):
        raise NotImplemented()


class FourStrokeCylinder(Cylinder):
    def __init__(self, in_source_position=True):
        self._position = 0
        if not in_source_position:
            self._position = 2
        self._subscribers = set()

    def subscribe(self, callback):
        self._subscribers.add(callback)

    def unsubscribe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _generate_event(self, event_text, **kwargs):
        for callback in self._subscribers:
            callback(event_text, **kwargs)

    def _sub_event(self, text):
        self._generate_event(text)
        self._generate_event('Поршень толкает коленвал')
        self._generate_event('Противовес стабилизирует направление вращения коленвала')
        self._generate_event('Коленвал вращает маховик')

    def process_stroke(self):

        if self._position == 0:
            self._generate_event('Начало впуска')
            self._sub_event('Поршень начинает опускаться с верхней мертвой точки')
            self._generate_event('Открывается впускной клапан', **{'fuel': None})
            self._generate_event('Создается разреженность')
            self._sub_event('Впрыскивается топливная жидкость')
            self._sub_event('Поршень достигает нижней мертвой точки')
            self._sub_event('Впусной клапан закрывается')
            self._generate_event('Завершение впуска')
        elif self._position == 1:
            self._generate_event('Начало сжатия')
            self._sub_event('Поршень начинает подниматься с нижней мертвой точки')
            self._sub_event('Горючее начинает сжиматься')
            self._sub_event('Поршень достигает верхней мертвой точки')
            self._generate_event('Завершение сжатия')
        elif self._position == 2:
            self._generate_event('Начало рабочего хода')
            self._generate_event('Свеча зажигания создает искру')
            self._generate_event('Топливо зажигается')
            self._generate_event('Создается большое давление')
            self._sub_event('Поршень начинает опускаться с верхней мертвой точки')
            self._sub_event('Поршень достигает нижней мертвой точки')
            self._generate_event('Завершение рабочего хода')
        elif self._position == 3:
            self._generate_event('Начало выпуска')
            self._generate_event('Открывается выпускной клапан')
            self._sub_event('Поршень начинает подниматься с нижней мертвой точки')
            self._sub_event('Отработанная смесь под давлением поршня вытесняется')
            self._sub_event('Поршень достигает верхней мертвой точки')
            self._generate_event('Закрывается выпускной клапан', **{'exhaust': None})
            self._generate_event('Завершение выпуска')
        self._position = (self._position + 1) % 4
