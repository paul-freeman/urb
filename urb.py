"""A basic interface to the Ultrasonic Relay Boxes"""
import sys
from time import sleep
from ast import literal_eval
from itertools import chain
from tkinter import filedialog
import tkinter as tk
from socket import socket as Socket
from socket import AF_INET, SOCK_STREAM, SHUT_RDWR, timeout
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d # pylint: disable=unused-import

# IP address of the oscilloscope
IP_ADDRESS = '130.216.55.153'

# This should be True if you are not triggering on the oscilloscope for some reason.
FORCE_TRIGGER = True

ARRANGE = None
WAVE = None
PRESSURE = None
SCALE = None
CANVAS = None
FIG = Figure(figsize=(16, 6))
DATA = {'P':[], 'S1':[], 'S2':[]}


class URBInterface(tk.Frame):  # pylint: disable=too-many-ancestors
    """A basic interface to the Ultrasonic Relay Boxes"""
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.src = UltrasonicRelayBox(ip_address='ultrasonic-src01')
        self.rcvr = UltrasonicRelayBox(ip_address='ultrasonic-rcvr01')
        self.scope = Oscilloscope(ip_address=IP_ADDRESS)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        """Generate the URB GUI."""
        self.create_stack_widgets()

        sub_frame = tk.Frame(self)
        for channel in self.scope.channel_list:
            channel.button = tk.Button(sub_frame)
            channel.button['text'] = 'Save: Ch{}'.format(channel.number)
            channel.button['command'] = channel.save
            channel.button.pack(side='left')

        self.scope.load = tk.Button(sub_frame)
        self.scope.load['text'] = 'Load'
        self.scope.load['command'] = self.scope.load_file
        self.scope.load.pack(side='left')

        self.scope.save = tk.Button(sub_frame)
        self.scope.save['text'] = 'Save'
        self.scope.save['command'] = self.scope.save_file
        self.scope.save.pack(side='left')
        sub_frame.pack(side='top')

        # Arrangement Radio Buttons
        sub_frame = tk.Frame(self)
        label = tk.Label(sub_frame)
        label.config(text='Arrange by: ')
        label.pack(side='left')
        global ARRANGE
        ARRANGE = tk.StringVar()
        ARRANGE.set('count')
        modes = [('Count', 'count'), ('Pressure', 'pressure'), ('Both', 'both')]
        for text, mode in modes:
            tk.Radiobutton(
                sub_frame,
                text=text,
                variable=ARRANGE,
                value=mode,
                command=draw_canvas).pack(side='left')
        sub_frame.pack(side='top')

        # Wave Radio Buttons
        sub_frame = tk.Frame(self)
        label = tk.Label(sub_frame)
        label.config(text='Wave type: ')
        label.pack(side='left')
        global WAVE
        WAVE = tk.StringVar()
        WAVE.set('P')
        modes = [('P wave', 'P'), ('S1 wave', 'S1'), ('S2 wave', 'S2')]
        for text, mode in modes:
            tk.Radiobutton(
                sub_frame,
                text=text,
                variable=WAVE,
                value=mode).pack(side='left')
        sub_frame.pack(side='top')

        # Pressure input
        sub_frame = tk.Frame(self)
        label = tk.Label(sub_frame)
        label.config(text='Pressure: ')
        label.pack(side='left')
        global PRESSURE
        PRESSURE = tk.StringVar()
        PRESSURE.set('0.0')
        tk.Entry(sub_frame, textvariable=PRESSURE).pack(side='left')
        sub_frame.pack(side='top')

        # Scale input
        sub_frame = tk.Frame(self)
        label = tk.Label(sub_frame)
        label.config(text='Scale voltage: ')
        label.pack(side='left')
        global SCALE
        SCALE = tk.StringVar()
        SCALE.set("1.0")
        tk.Entry(sub_frame, textvariable=SCALE).pack(side='left')
        button = tk.Button(sub_frame, text='Update', command=draw_canvas).pack(side='left')
        sub_frame.pack(side='top')

        self.create_canvas_widget()
        draw_canvas()

    def create_canvas_widget(self):
        """Create canvas on which to draw plots"""
        global CANVAS
        CANVAS = FigureCanvasTkAgg(FIG, master=self)
        CANVAS.get_tk_widget().pack(side='top')

    def create_stack_widgets(self):
        """Generate the buttons for the P, S1, S2 waves."""
        for stack in chain(self.src.stack_list, self.rcvr.stack_list):
            sub_frame = tk.Frame(self)
            label = tk.Label(sub_frame)
            label.config(text='{} URB: '.format(stack.get_type()))
            label.pack(side='left')
            if stack.connected:
                backgroud_color = 'red'
            else:
                backgroud_color = 'grey'
            # P button
            stack.button_dict['P'] = tk.Button(sub_frame, fg='white', bg=backgroud_color)
            stack.button_dict['P']['text'] = 'P wave {}'.format(stack.stacknum)
            if stack.connected:
                stack.button_dict['P']['command'] = stack.toggle_p
            stack.button_dict['P'].pack(side='left')
            # S1 button
            stack.button_dict['S1'] = tk.Button(sub_frame, fg='white', bg=backgroud_color)
            stack.button_dict['S1']['text'] = 'S1 wave {}'.format(stack.stacknum)
            if stack.connected:
                stack.button_dict['S1']['command'] = stack.toggle_s1
            stack.button_dict['S1'].pack(side='left')
            # S2 button
            stack.button_dict['S2'] = tk.Button(sub_frame, fg='white', bg=backgroud_color)
            stack.button_dict['S2']['text'] = 'S2 wave {}'.format(stack.stacknum)
            if stack.connected:
                stack.button_dict['S2']['command'] = stack.toggle_s2
            stack.button_dict['S2'].pack(side='left')
            sub_frame.pack(side='top')

def draw_canvas():
    """Draw the plotting canvas"""
    cmap = cm.get_cmap('viridis')
    title = {'P':'P waves', 'S1':'S1 waves', 'S2':'S2 waves'}
    FIG.clear()
    for i, (wave_type, data) in enumerate(DATA.items()):
        if ARRANGE.get() == 'both':
            ax = FIG.add_subplot(1, 3, i+1, projection='3d')
            ax.clear()
        else:
            ax = FIG.add_subplot(1, 3, i+1)
            ax.clear()
        ax.set_title(title[wave_type])
        for count, (pressure, trace) in enumerate(reversed(data)):
            sample = len(data) - count # because reversed
            if len(data) <= 1:
                value = 0.0
            else:
                value = (sample-1)/(len(data)-1)
            if ARRANGE.get() == 'count':
                ax.plot(trace[0], trace[1] * float(SCALE.get()) + sample, color=cmap(value))
            elif ARRANGE.get() == 'pressure':
                ax.plot(trace[0], trace[1] * float(SCALE.get()) + pressure, color=cmap(value))
            elif ARRANGE.get() == 'both':
                ax.plot(trace[0], [sample] * len(trace[0]), trace[1] * float(SCALE.get()) + pressure, color=cmap(value))
    CANVAS.draw()

class Oscilloscope():
    """The Tektronix Oscilloscope"""
    def __init__(self, ip_address, port=4000, channels=4):
        self.ip_address = ip_address
        self.port = port
        self.channel_list = [Channel(channelnum + 1, self) for channelnum in range(channels)]
        self.button_list = [None] * channels
        self.load = None
        self.save = None
        self.data = None

    def load_file(self):
        """Load data from disk"""
        filename = filedialog.askopenfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            title='Select CSV file to load')
        if not filename:
            return
        data = np.loadtxt(
            filename,
            dtype=[('wave', 'S5'),
                   ('sample', '<i4'),
                   ('pressure', '<f4'),
                   ('time', '<f4'),
                   ('voltage', '<f4')],
            delimiter=',',
            skiprows=1)
        global DATA
        DATA = {'P':[], 'S1':[], 'S2':[]}
        for wave_type, sample, pressure, time, volt in data:
            wave_type = literal_eval(wave_type.decode()).decode()
            try:
                curr_pressure, trace = DATA[wave_type][sample]
                if curr_pressure != pressure:
                    raise ValueError('mismatched pressures in data')
                trace.append((time, volt))
            except IndexError:
                DATA[wave_type].append((pressure, [(time, volt)]))
        for wave_type in DATA:
            for i, (pressure, trace) in enumerate(DATA[wave_type]):
                trace = np.asarray(trace).T
                time = np.asarray(trace[0])
                volts = np.asarray(trace[1])
                DATA[wave_type][i] = (pressure, np.array([time, volts]))
        print(DATA)
        draw_canvas()
            
    def save_file(self):
        """Save data to disk"""
        filename = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile='wave_data.csv',
            title='Select output CSV file')
        if not filename:
            return
        data = []
        for wave_type, waves in DATA.items():
            for sample, (pressure, trace) in enumerate(waves):
                for time, volt in trace.T:
                    data.append((wave_type, sample, pressure, time, volt))
        save_data = np.asarray(data, dtype=[
            ('wave', 'S2'),
            ('sample', '<i4'),
            ('pressure', '<f4'),
            ('time', '<f4'),
            ('voltage', '<f4')])
        np.savetxt(filename, save_data, fmt='%s,%i,%.4f,%.6f,%.6f',
            header='Wave,Count,Pressure,Time,Voltage')

class Channel():
    """A channel on the oscilloscope"""
    def __init__(self, number, scope):
        self.number = number
        self.scope = scope

    def save(self):
        """Save a trace into the data"""
        name = 'DPO3014'
        # execute the PLACE code to get data from the oscilloscope
        config = {"force_trigger": FORCE_TRIGGER}
        metadata = {}
        instrument = DPO3014(config)
        try:
            instrument.config(metadata, 1)
            waveform = instrument.update()[name + '-trace'][0][self.number-1]
            time, volts = self.transform_data(waveform, metadata)
            DATA[WAVE.get()].append((float(PRESSURE.get()), np.array([time, volts])))
            draw_canvas()
        except OSError:
            print('connection to {} timed out'.format(IP_ADDRESS))

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
    def __init__(self, ip_address='192.168.0.0', port=9876, stacks=1):
        self.ip_address = ip_address
        self.port = port
        self.stack_list = [Stack(stacknum + 1, self) for stacknum in range(stacks)]
        self.type = 'Unknown'
        self.clear()  # cannot use this during testing

    def clear(self):
        """Set all relays in this URB to the 'None' state."""
        with Socket(AF_INET, SOCK_STREAM) as socket:
            socket.settimeout(5)
            try:
                socket.connect((self.ip_address, self.port))
            except timeout:
                print('connection to {} timed out'.format(self.ip_address))
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
            print('turn {} off'.format(wave))
            self._mode('None')
            self.button_dict[wave]['fg'] = 'white'
            self.button_dict[wave]['bg'] = 'red'
            return
        # otherwise, toggle the current mode and turn on the new mode
        if self.mode != 'None':
            self._toggle(self.mode)
        print('turn {} on'.format(wave))
        self._mode(wave)
        self.button_dict[wave]['fg'] = 'black'
        self.button_dict[wave]['bg'] = 'green'

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
        self._ip_address = IP_ADDRESS
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
        self._activate_acquisition()
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
    matplotlib.use('TkAgg')
    ROOT = tk.Tk()
    APP = URBInterface(master=ROOT)
    APP.mainloop()
