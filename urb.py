"""A basic interface to the Ultrasonic Relay Boxes"""
import sys
import os.path
import glob
from time import sleep
from functools import partial
from ast import literal_eval
from tkinter import filedialog
import tkinter as tk
from socket import socket as Socket
from socket import AF_INET, SOCK_STREAM, SHUT_RDWR, timeout
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# pylint: disable=wrong-import-position
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d # pylint: disable=unused-import

OSCILLOSCOPE_IP_ADDRESS = '130.216.57.190'
OSCILLOSCOPE_PORT = 4000
OSCILLOSCOPE_FORCE = True
CHANNELS = 4

URB_SRC_IP_ADDRESS = 'ultrasonic-src01'
URB_RCV_IP_ADDRESS = 'ultrasonic-rcvr01'
URB_PORT = 9876

STACK_ACTIVE_COLOR = 'green', 'white'
STACK_CONNECTED_COLOR = 'red', 'white'
STACK_DISCONNECTED_COLOR = 'grey', 'white'


class URBInterface(tk.Frame): # pylint: disable=too-many-ancestors
    """A basic interface to the Ultrasonic Relay Boxes"""
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.fig = Figure(figsize=(16, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        NavigationToolbar2TkAgg(self.canvas, self)
        self.config = {'arrange': tk.StringVar(),
                       'wave': tk.StringVar(),
                       'pressure': tk.StringVar(),
                       'scale': tk.StringVar(),
                       'lag': tk.StringVar(),
                       'note': tk.StringVar()}
        self.data = {'P':[], 'S1':[], 'S2':[]}
        self.src = UltrasonicRelayBox(ip_address=URB_SRC_IP_ADDRESS, port=URB_PORT)
        self.rcvr = UltrasonicRelayBox(ip_address=URB_RCV_IP_ADDRESS, port=URB_PORT)
        self.channel_list = [Channel(channelnum + 1) for channelnum in range(CHANNELS)]
        self.pack()
        self.create_widgets()


    def create_widgets(self):
        """Generate the URB GUI."""
        # Menu items
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Load NPY file...', command=self.load_file)
        filemenu.add_command(label='Save NPY file...', command=self.save_file)
        filemenu.add_command(label='Load CSVs from directory...', command=self.load_folder)
        filemenu.add_command(label='Save CSVs to directory...', command=self.save_folder)
        menubar.add_cascade(label='File', menu=filemenu)
        self.master.config(menu=menubar)
        # URB Wave Buttons
        for num, (src, rcv) in enumerate(zip(self.src.stack_list, self.rcvr.stack_list)):
            URBFrame(self, src, rcv, num).pack(side='top')
        # Wave/Pressure/TimeLag input
        self.make_input_frame()
        # Save Buttons
        frame = tk.Frame(self)
        for wave_type in sorted(self.data):
            sub_frame = tk.Frame(frame)
            tk.Label(sub_frame, text='Save {}: '.format(wave_type)).pack(side='left')
            for channel in self.channel_list:
                channel.button = tk.Button(sub_frame)
                channel.button['text'] = 'Ch {}'.format(channel.number)
                cmd = partial(self.save, channel, wave_type)
                channel.button['command'] = cmd
                channel.button.pack(side='left')
            sub_frame.pack(side='left', padx=90)
        frame.pack(side='top')
        # Plots
        self.canvas.get_tk_widget().pack(side='top')
        # Arrangement Radio Buttons
        sub_frame = tk.Frame(self)
        tk.Label(sub_frame, text='Plot type:').pack(side='left')
        self.config['arrange'].set('pressure')
        modes = [('Last', 'last'),
                 ('Count', 'count'),
                 ('Pressure', 'pressure'),
                 ('Overlay', 'overlay'),
                 ('3d', '3d')]
        for text, mode in modes:
            tk.Radiobutton(
                sub_frame,
                text=text,
                variable=self.config['arrange'],
                value=mode,
                command=self.draw_canvas).pack(side='left')
        sub_frame.pack(side='top')
        # Scale input
        sub_frame = tk.Frame(self)
        label = tk.Label(sub_frame)
        label.config(text='Scale voltage: ')
        label.pack(side='left')
        self.config['scale'].set("1.0")
        tk.Entry(sub_frame, textvariable=self.config['scale']).pack(side='left')
        tk.Button(sub_frame, text='Update', command=self.draw_canvas).pack(side='left')
        sub_frame.pack(side='top')
        # Draw
        self.draw_canvas()


    def make_input_frame(self):
        """Create the sub frame for the input values"""
        sub_frame = tk.Frame(self)

        # Pressure Entry
        sub_sub_frame = tk.Frame(sub_frame, padx=10)
        tk.Label(sub_sub_frame, text='Pressure: ').pack(side='left')
        self.config['pressure'].set('0.0')
        tk.Entry(sub_sub_frame, textvariable=self.config['pressure']).pack(side='left')
        sub_sub_frame.pack(side='left')

        # Lag Entry
        sub_sub_frame = tk.Frame(sub_frame, padx=10)
        tk.Label(sub_sub_frame, text='Time lag: ').pack(side='left')
        self.config['lag'].set('0.0')
        tk.Entry(sub_sub_frame, textvariable=self.config['lag']).pack(side='left')
        sub_sub_frame.pack(side='left')

        # Note Entry
        sub_sub_frame = tk.Frame(sub_frame, padx=10)
        tk.Label(sub_sub_frame, text='Note: ').pack(side='left')
        self.config['note'].set('')
        tk.Entry(sub_sub_frame, textvariable=self.config['note']).pack(side='left')
        sub_sub_frame.pack(side='left')

        sub_frame.pack(side='top')


    def save(self, channel, wave_type):
        """Call save on the requested channel and redraw"""
        self.config['wave'].set(wave_type)
        channel.save(self.data, self.config)
        self.draw_canvas()


    def draw_canvas(self):
        """Draw the plotting canvas"""
        cmap = cm.get_cmap('viridis')
        title = {'P':'P waves', 'S1':'S1 waves', 'S2':'S2 waves'}
        self.fig.clear()
        axes_list = []
        for i, (wave_type, data) in enumerate(sorted(self.data.items())):
            if self.config['arrange'].get() == '3d':
                axes = self.fig.add_subplot(1, 3, i+1, projection='3d')
            else:
                axes = self.fig.add_subplot(1, 3, i+1)
            axes.clear()
            axes_list.append(axes)
            axes.set_title(title[wave_type])
            for count, maybe_data in enumerate(reversed(data)):
                if not maybe_data:
                    continue
                pressure, lag, _, trace = maybe_data
                sample = len(data) - count # because reversed
                if len(data) <= 1:
                    value = 0.0
                else:
                    value = (sample-1)/(len(data)-1)
                arrange = self.config['arrange']
                scale = self.config['scale']

                if arrange.get() == 'last':
                    # plot last trace
                    axes.plot(trace[0] - lag,
                              trace[1] * float(scale.get()))
                    break

                elif arrange.get() == 'count':
                    # plot traces offset by count
                    try:
                        axes.plot(trace[0] - lag,
                                  trace[1] * float(scale.get()) + sample,
                                  color=cmap(value))
                    except (TypeError, IndexError):
                        print('maybe_data = {}'.format(maybe_data))
                        print('type(trace[0]) = {}'.format(type(trace[0])))
                        print('type(trace[1]) = {}'.format(type(trace[1])))
                        print('type(lag) = {}'.format(type(lag)))
                        raise

                elif arrange.get() == 'pressure':
                    # plot traces offset by pressure
                    axes.plot(trace[0] - lag,
                              trace[1] * float(scale.get()) + pressure,
                              color=cmap(value))

                elif arrange.get() == 'overlay':
                    # plot overlayed traces
                    try:
                        axes.plot(trace[0] - lag,
                                  trace[1] * float(scale.get()),
                                  color=cmap(value))
                    except (TypeError, IndexError):
                        print('maybe_data = {}'.format(maybe_data))
                        print('type(trace[0]) = {}'.format(type(trace[0])))
                        print('type(trace[1]) = {}'.format(type(trace[1])))
                        print('type(lag) = {}'.format(type(lag)))
                        raise

                elif arrange.get() == '3d':
                    # plot 3D traces offset by pressure and count
                    try:
                        axes.plot(trace[0] - lag,
                                  [sample] * len(trace[0]),
                                  trace[1] * float(scale.get()) + pressure,
                                  color=cmap(value))
                    except TypeError:
                        print('maybe_data = {}'.format(maybe_data))
                        print('type(trace[0]) = {}'.format(type(trace[0])))
                        print('type(lag) = {}'.format(type(lag)))
                        raise

        y_min = min(axes.get_ylim()[0] for axes in axes_list)
        y_max = max(axes.get_ylim()[1] for axes in axes_list)
        _ = [axes.set_ylim(y_min, y_max, auto=True) for axes in axes_list]
        self.canvas.draw()


    def load_file(self):
        """Load NPY data from disk"""
        filename = filedialog.askopenfilename(
            defaultextension='.npy',
            filetypes=[('NPY', '*.npy')],
            title='Select file to load')
        if not filename:
            return
        with open(filename, 'rb') as f_in:
            self.data = np.load(f_in).tolist()
        self.draw_canvas()


    def load_folder(self):
        """Load CSV data from separate files in a folder"""
        self.data = {'P':[], 'S1':[], 'S2':[]}
        for path in glob.iglob(os.path.join(filedialog.askdirectory(), 'Wave_*.csv')):
            values = os.path.splitext(os.path.split(path)[1])[0].split('_')
            wave_type = values[1]
            sample = literal_eval(values[2][len('Num'):])
            pressure = literal_eval(values[3][len('Press'):])
            lag = literal_eval(values[4][len('Lag'):])
            try:
                note = values[5]
            except IndexError:
                note = ''
            raw_data = np.loadtxt(path, delimiter=',', skiprows=1)
            while True:
                try:
                    self.data[wave_type][sample] = (
                        pressure, lag, note, np.reshape(raw_data.T, (2, -1))
                    )
                    break
                except IndexError:
                    self.data[wave_type].append(None)
        self.draw_canvas()


    def save_file(self):
        """Save NPY data to disk"""
        filename = filedialog.asksaveasfilename(
            defaultextension='.npy',
            filetypes=[('NPY', '*.npy')],
            initialfile='wave_data.npy',
            title='Select output file')
        with open(filename, 'xb') as f_out:
            np.save(f_out, self.data)


    def save_folder(self):
        """Save CSV data to separate files in a directory"""
        directory = filedialog.askdirectory()
        if not os.path.exists(directory):
            os.makedirs(directory)
        for wave_type, waves in self.data.items():
            for sample, maybe_data in enumerate(waves):
                if not maybe_data:
                    continue
                pressure, lag, note, trace = maybe_data
                if note == '':
                    filename = 'Wave_{}_Num{:d}_Press{:d}_Lag{:.3f}.csv'.format(
                        wave_type, int(sample), int(pressure), lag)
                else:
                    valid = ' _-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    note = ''.join(['-' if c in ' _' else c for c in note[12:] if c in valid])
                    filename = 'Wave_{}_Num{:d}_Press{:d}_Lag{:.3f}_{}.csv'.format(
                        wave_type, int(sample), int(pressure), lag, note)
                data = []
                for time, volt in trace.T:
                    data.append((time, volt))
                save_data = np.asarray(data, dtype=[('time', '<f4'), ('voltage', '<f4')])
                path = os.path.join(directory, filename)
                np.savetxt(path, save_data, fmt='%.6f,%.6f', header='Time,Voltage')


class URBFrame(tk.Frame): # pylint: disable=too-many-ancestors
    """Defines a Frame to describe a URB"""
    def __init__(self, master, src, rcv, num):
        tk.Frame.__init__(self, master)
        tk.Label(self, text='URB {}: '.format(num)).pack(side='left')
        StackFrame(self, src).pack(side='left', padx=10)
        StackFrame(self, rcv).pack(side='left', padx=10)


class StackFrame(tk.Frame): # pylint: disable=too-many-ancestors
    """Defines a Frame to describe a stack"""
    def __init__(self, master, stack):
        tk.Frame.__init__(self, master)
        tk.Label(self, text='{}: '.format(stack.get_type())).pack(side='left')
        if stack.connected:
            bg_value, fg_value = STACK_CONNECTED_COLOR
        else:
            bg_value, fg_value = STACK_DISCONNECTED_COLOR
        # P button
        stack.button_dict['P'] = tk.Button(self, fg=fg_value, bg=bg_value)
        stack.button_dict['P']['text'] = 'P wave {}'.format(stack.stacknum)
        if stack.connected:
            stack.button_dict['P']['command'] = stack.toggle_p
        stack.button_dict['P'].pack(side='left')
        # S1 button
        stack.button_dict['S1'] = tk.Button(self, fg=fg_value, bg=bg_value)
        stack.button_dict['S1']['text'] = 'S1 wave {}'.format(stack.stacknum)
        if stack.connected:
            stack.button_dict['S1']['command'] = stack.toggle_s1
        stack.button_dict['S1'].pack(side='left')
        # S2 button
        stack.button_dict['S2'] = tk.Button(self, fg=fg_value, bg=bg_value)
        stack.button_dict['S2']['text'] = 'S2 wave {}'.format(stack.stacknum)
        if stack.connected:
            stack.button_dict['S2']['command'] = stack.toggle_s2
        stack.button_dict['S2'].pack(side='left')


class Channel():
    """A channel on the oscilloscope"""
    def __init__(self, number):
        self.number = number


    def save(self, data, settings):
        """Save a trace into the data"""
        name = 'DPO3014'
        # execute the PLACE code to get data from the oscilloscope
        config = {"force_trigger": OSCILLOSCOPE_FORCE}
        metadata = {}
        instrument = DPO3014(config)
        try:
            instrument.config(metadata, 1)
            waveform = instrument.update()[name + '-trace'][0][self.number-1]
            time, volts = self.transform_data(waveform, metadata)
            data[settings['wave'].get()].append(
                (float(settings['pressure'].get()),
                 float(settings['lag'].get()),
                 np.array([time, volts])))
        except OSError:
            print('connection to {} timed out'.format(OSCILLOSCOPE_IP_ADDRESS))


    def transform_data(self, waveform, metadata):
        """convert waveform to proper values"""
        print('performing calculations... ', end='')
        name = 'DPO3014'
        sys.stdout.flush()
        xzero = metadata[name + '-ch{:d}_x_zero'.format(self.number)]
        xincr = metadata[name + '-ch{:d}_x_increment'.format(self.number)]
        yzero = metadata[name + '-ch{:d}_y_zero'.format(self.number)]
        yoff = metadata[name + '-ch{:d}_y_offset'.format(self.number)]
        ymult = metadata[name + '-ch{:d}_y_multiplier'.format(self.number)]

        #Transform to Volts
        volts = (waveform - yoff) * ymult  + yzero
        time = 1e6*(np.linspace(0, xincr * len(volts), len(volts))+xzero)
        print('done')
        return time, volts


class UltrasonicRelayBox():
    """An Ultrasonic Relay Box"""
    def __init__(self, ip_address, port, stacks=1):
        self.ip_address = ip_address
        self.port = port
        self.stack_list = [Stack(stacknum + 1, self) for stacknum in range(stacks)]
        self.type = 'Unknown'
        self.clear()  # cannot use this during testing


    def get_stack_number(self):
        """Return the number of stacks in the relay box"""
        return len(self.stack_list)


    def clear(self):
        """Set all relays in this URB to the 'None' state."""
        with Socket(AF_INET, SOCK_STREAM) as socket:
            socket.settimeout(5)
            try:
                socket.connect((self.ip_address, self.port))
            except timeout:
                print('connection to {} timed out'.format(self.ip_address))
                return
            except OSError:
                print('could not find path to {}'.format(self.ip_address))
                return
            socket.send(bytes('Clear\r\n', encoding='ascii'))
            query = literal_eval(recv_end(socket))
            self.type = query['type']
            socket.shutdown(SHUT_RDWR)
            for stack in self.stack_list:
                stack.connected = True


class Stack():
    """A stack on an Ultrasonic Relay Box"""
    def __init__(self, stacknum, urb):
        self.stacknum = stacknum
        self.urb = urb
        self.mode = 'None'
        self.button_dict = {'P': None, 'S1': None, 'S2': None}
        self.connected = False


    def get_type(self):
        """Return the URB type."""
        return self.urb.type


    def toggle_p(self):
        """Toggle the P wave on or off."""
        self._toggle('P')


    def toggle_s1(self):
        """Toggle the S1 wave on or off."""
        self._toggle('S1')


    def toggle_s2(self):
        """Toggle the S2 wave on or off."""
        self._toggle('S2')


    def _mode(self, wave):
        with Socket(AF_INET, SOCK_STREAM) as socket:
            socket.settimeout(5)
            socket.connect((self.urb.ip_address, self.urb.port))
            socket.send(bytes("Wave {} {}\r\n".format(self.stacknum, wave), encoding='ascii'))
            query = literal_eval(recv_end(socket))
            socket.shutdown(SHUT_RDWR)
        self.mode = query['mode']


    def _toggle(self, wave):
        # if the current mode is the new mode, set wave to none
        if self.mode == wave:
            bg_value, fg_value = STACK_CONNECTED_COLOR
            print('turn {} off'.format(wave))
            self._mode('None')
            self.button_dict[wave]['fg'] = fg_value
            self.button_dict[wave]['bg'] = bg_value
            return
        # otherwise, toggle the current mode and turn on the new mode
        if self.mode != 'None':
            self._toggle(self.mode)
        print('turn {} on'.format(wave))
        self._mode(wave)
        bg_value, fg_value = STACK_ACTIVE_COLOR
        self.button_dict[wave]['fg'] = fg_value
        self.button_dict[wave]['bg'] = bg_value


def recv_end(socket):
    """Receive data until the line termination is seen."""
    total_data = ''
    while True:
        data = socket.recv(8192).decode('ascii')
        if data[-2:] == '\r\n':
            total_data += data[:-2]
            break
        if data[-1:] == '\r':  # the \r\n was split
            total_data += data[:-1]
            # need to read the \n
            data = socket.recv(1).decode('ascii')
            break
        total_data += data
    return total_data


class MSO3000andDPO3000Series():
    #pylint: disable=too-many-instance-attributes
    """PLACE device class for the MSO3000 and DPO3000 series oscilloscopes."""
    _bytes_per_sample = 2
    _data_type = np.dtype('<i'+str(_bytes_per_sample)) # (<)little-endian, (i)signed integer


    def __init__(self, config):
        self._config = config
        self.priority = 100
        self._updates = None
        self._ip_address = None
        self._scope = None
        self._channels = None
        self._samples = None
        self._record_length = None
        self._x_zero = None
        self._x_increment = None


    def config(self, metadata, total_updates):
        """Configure the oscilloscope.

        :param metadata: metadata for the experiment
        :type metadata: dict

        :param total_updates: the number of update steps that will be in this experiment
        :type total_updates: int

        :raises OSError: if unable to connect to oscilloscope
        """
        name = self.__class__.__name__
        self._updates = total_updates
        self._ip_address = OSCILLOSCOPE_IP_ADDRESS
        self._scope = Socket(AF_INET, SOCK_STREAM)
        self._scope.settimeout(5.0)
        try:
            self._scope.connect((self._ip_address, 4000))
        except OSError:
            self._scope.close()
            del self._scope
            raise
        self._channels = [self._is_active(x+1) for x in range(self._get_num_analog_channels())]
        self._record_length = self._get_record_length()
        metadata[name + '-record_length'] = self._record_length
        self._x_zero = [None for _ in self._channels]
        self._x_increment = [None for _ in self._channels]
        metadata[name + '-active_channels'] = self._channels
        self._samples = self._get_sample_rate()
        metadata[name + '-sample_rate'] = self._samples
        for channel, active in enumerate(self._channels):
            if not active:
                continue
            chan = channel+1
            self._send_config_msg(chan)
            self._x_zero[channel] = self._get_x_zero(chan)
            self._x_increment[channel] = self._get_x_increment(chan)
            metadata[name + '-ch{:d}_x_zero'.format(chan)] = self._x_zero[channel]
            metadata[name + '-ch{:d}_x_increment'.format(chan)] = self._x_increment[channel]
            metadata[name + '-ch{:d}_y_zero'.format(chan)] = self._get_y_zero(chan)
            metadata[name + '-ch{:d}_y_offset'.format(chan)] = self._get_y_offset(chan)
            metadata[name + '-ch{:d}_y_multiplier'.format(chan)] = self._get_y_multiplier(chan)
        self._scope.close()


    def update(self):
        """Get data from the oscilloscope.

        :returns: the trace data
        :rtype: numpy.array dtype='(*number_channels*,*number_samples*)int16'
        """
        self._scope = Socket(AF_INET, SOCK_STREAM)
        self._scope.settimeout(5.0)
        self._scope.connect((self._ip_address, 4000))
        #self._activate_acquisition()
        field = '{}-trace'.format(self.__class__.__name__)
        type_ = '({:d},{:d})int16'.format(len(self._channels), self._record_length)
        data = np.zeros((1,), dtype=[(field, type_)])
        print('transfering waveform... ', end='')
        sys.stdout.flush()
        for channel, active in enumerate(self._channels):
            if not active:
                continue
            self._request_curve(channel+1)
            trace = self._receive_curve()
            data[field][0][channel] = trace
        self._scope.close()
        print('done')
        return data.copy()


    def cleanup(self, abort=False):
        """End the experiment.

        :param abort: indicates the experiment is being aborted rather than
                      having finished normally
        :type abort: bool
        """
        pass


    def _clear_errors(self):
        self._scope.sendall(bytes(':*ESR?;:ALLEv?\n', encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')


    def _is_active(self, channel):
        self._scope.settimeout(5.0)
        self._clear_errors()
        self._scope.sendall(bytes(':DATA:SOURCE CH{:d};:WFMOUTPRE?\n'.format(channel),
                                  encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        self._scope.sendall(b'*ESR?\n')
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        self._clear_errors()
        return int(dat) == 0


    def _get_num_analog_channels(self):
        self._scope.settimeout(5.0)
        self._scope.sendall(b':CONFIGURATION:ANALOG:NUMCHANNELS?\n')
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return int(dat)


    def _get_x_zero(self, channel):
        self._scope.settimeout(5.0)
        self._scope.sendall(bytes(
            ':HEADER OFF;:DATA:SOURCE CH{:d};:WFMOUTPRE:XZERO?\n'.format(channel),
            encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return float(dat)


    def _get_y_zero(self, channel):
        self._scope.settimeout(5.0)
        self._scope.sendall(bytes(
            ':HEADER OFF;:DATA:SOURCE CH{:d};:WFMOUTPRE:YZERO?\n'.format(channel),
            encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return float(dat)


    def _get_x_increment(self, channel):
        self._scope.settimeout(5.0)
        self._scope.sendall(bytes(
            ':HEADER OFF;:DATA:SOURCE CH{:d};:WFMOUTPRE:XINCR?\n'.format(channel),
            encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return float(dat)


    def _get_y_offset(self, channel):
        self._scope.settimeout(5.0)
        self._scope.sendall(bytes(
            ':HEADER OFF;:DATA:SOURCE CH{:d};:WFMOUTPRE:YOFF?\n'.format(channel),
            encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return float(dat)


    def _get_y_multiplier(self, channel):
        self._scope.settimeout(5.0)
        self._scope.sendall(bytes(
            ':HEADER OFF;:DATA:SOURCE CH{:d};:WFMOUTPRE:YMULT?\n'.format(channel),
            encoding='ascii'))
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return float(dat)


    def _get_sample_rate(self):
        self._scope.settimeout(5.0)
        self._scope.sendall(b':HEADER OFF;:HORIZONTAL:SAMPLERATE?\n')
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return float(dat)


    def _get_record_length(self):
        self._scope.settimeout(5.0)
        self._scope.sendall(b':HEADER OFF;:HORIZONTAL:RECORDLENGTH?\n')
        dat = ''
        while '\n' not in dat:
            dat += self._scope.recv(4096).decode('ascii')
        return int(dat)


    def _send_config_msg(self, channel):
        config_msg = bytes(
            ':DATA:' + (
                'SOURCE CH{:d};'.format(channel) +
                'START 1;' +
                'STOP {};'.format(self._record_length)
            ) +
            ':WFMOUTPRE:' + (
                'BYT_NR 2;' +
                'BIT_NR 16;' +
                'ENCDG BINARY;' +
                'BN_FMT RI;' +
                'BYT_OR LSB;'
            ) +
            ':HEADER 0\n',
            encoding='ascii'
        )
        self._scope.sendall(config_msg)


    def _activate_acquisition(self):
        self._scope.sendall(b':ACQUIRE:STATE ON\n')
        sleep(0.1)
        if self._config['force_trigger']:
            self._force_trigger()
        else:
            self._wait_for_trigger()


    def _force_trigger(self):
        for _ in range(120):
            self._scope.settimeout(60)
            self._scope.sendall(b':TRIGGER FORCE\n')
            sleep(0.1)
            self._scope.settimeout(0.25)
            try:
                self._scope.recv(4096)
            except OSError:
                pass
            self._scope.settimeout(60)
            self._scope.sendall(b':ACQUIRE:STATE?\n')
            sleep(0.1)
            byte = b''
            for _ in range(600):
                byte = self._scope.recv(1)
                if byte == b'0' or byte == b'1':
                    self._scope.settimeout(0.25)
                    try:
                        self._scope.recv(4096)
                    except OSError:
                        pass
                    break
            if byte == b'0':
                break


    def _wait_for_trigger(self):
        print('waiting for trigger(s)... ', end='')
        sys.stdout.flush()
        self._scope.setblocking(False)
        for _ in range(120):
            self._scope.sendall(b':ACQUIRE:STATE?\n')
            byte = b''
            for _ in range(600):
                try:
                    byte = self._scope.recv(1)
                except BlockingIOError:
                    sleep(0.1)
                    continue
                if byte == b'0' or byte == b'1':
                    break

            if byte == b'0':
                break
            sleep(0.5)
        print('done')
        sys.stdout.flush()


    def _request_curve(self, channel):
        self._scope.settimeout(60.0)
        self._scope.sendall(
            bytes(':DATA:SOURCE CH{:d};:CURVE?\n'.format(channel), encoding='ascii'))


    def _receive_curve(self):
        hash_message = b''
        while hash_message != b'#':
            hash_message = self._scope.recv(1)

        length_length = int(self._scope.recv(1).decode(), base=16)
        length = int(self._scope.recv(length_length).decode(), base=10)
        data = b''
        while len(data) < length:
            data += self._scope.recv(4096)
        data = data[:length]
        return np.frombuffer(data, dtype='int16')


class DPO3014(MSO3000andDPO3000Series):
    """Subclass for the DPO3014"""
    pass


if __name__ == "__main__":
    ROOT = tk.Tk()
    APP = URBInterface(master=ROOT)
    APP.mainloop()
