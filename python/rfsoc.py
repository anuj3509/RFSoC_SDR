from backend import *
from backend import be_np as np, be_scp as scipy
from signal_utilsrfsoc import Signal_Utils_Rfsoc
from tcp_comm import Tcp_Comm_RFSoC
try:
    from Sivers.siversController import *
except Exception as e:
    print("Error importing siversController class: ", e)




class RFSoC(Signal_Utils_Rfsoc):
    def __init__(self, params):
        super().__init__(params)

        self.beam_test = params.beam_test
        self.project = params.project
        self.board = params.board
        self.RFFE = params.RFFE
        self.TCP_port_Cmd = params.TCP_port_Cmd
        self.TCP_port_Data = params.TCP_port_Data
        self.lmk_freq_mhz = params.lmk_freq_mhz
        self.lmx_freq_mhz = params.lmx_freq_mhz
        self.dac_fs = params.fs_tx
        self.adc_fs = params.fs_rx
        self.bit_file_path = params.bit_file_path
        self.mix_freq_dac = params.mix_freq_dac
        self.mix_freq_adc = params.mix_freq_adc
        self.mix_phase_off = params.mix_phase_off
        self.DynamicPLLConfig = params.DynamicPLLConfig
        self.do_mixer_settings = params.do_mixer_settings
        self.do_pll_settings = params.do_pll_settings
        self.run_tcp_server = params.run_tcp_server
        self.verbose_level = params.verbose_level
        self.n_frame_wr=params.n_frame_wr
        self.n_frame_rd=params.n_frame_rd
        self.n_tx_ant = params.n_tx_ant
        self.n_rx_ant = params.n_rx_ant
        self.params = params # Store params for later use
        
        if self.board=='rfsoc_2x2':
            self.adc_bits = 12
            self.dac_bits = 14
            self.adc_max_fs = 4096e6
            self.dac_max_fs = 6554e6
        elif self.board=='rfsoc_4x2':
            self.adc_bits = 14
            self.dac_bits = 14
            self.adc_max_fs = 5000e6
            self.dac_max_fs = 9850e6

        if 'ddr4' in self.project:
            if self.board=='rfsoc_2x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 1:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = [0,1]
                    self.adc_tiles_sync = [0,2]
                elif 'sounder_if' in self.project:
                    if self.n_tx_ant==1:
                        self.dac_tile_block_dic = {1: [0]}
                        self.dac_tiles_sync = [0]
                    elif self.n_tx_ant==2:
                        self.dac_tile_block_dic = {0: [0], 1:[0]}
                        self.dac_tiles_sync = [0,1]
                    if self.n_rx_ant==1:
                        self.adc_tile_block_dic = {2: [0]}
                        self.adc_tiles_sync = [0,2]
                    elif self.n_rx_ant==2:
                        self.adc_tile_block_dic = {0: [0], 2:[0]}
                        self.adc_tiles_sync = [0,2]
            elif self.board=='rfsoc_4x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 2:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = [0,2]
                    self.adc_tiles_sync = [0,2]
                elif 'sounder_if' in self.project:
                    if self.n_tx_ant==1:
                        self.dac_tile_block_dic = {2: [0]}
                        self.dac_tiles_sync = []
                    elif self.n_tx_ant==2:
                        self.dac_tile_block_dic = {0: [0], 2:[0]}
                        self.dac_tiles_sync = []
                    if self.n_rx_ant==1:
                        self.adc_tile_block_dic = {2: [0]}
                        self.adc_tiles_sync = [0,2]
                    elif self.n_rx_ant==2:
                        self.adc_tile_block_dic = {0: [0], 2:[0]}
                        self.adc_tiles_sync = [0,2]
                    
        else:
            if self.board=='rfsoc_2x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 1:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []
                elif 'sounder_if' in self.project:
                    self.dac_tile_block_dic = {1: [0]}
                    self.adc_tile_block_dic = {2: [0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []
            elif self.board=='rfsoc_4x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 2:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []
                elif 'sounder_if' in self.project:
                    self.dac_tile_block_dic = {2: [0]}
                    self.adc_tile_block_dic = {2: [0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []

        if 'ddr4' in self.project:
            if 'sounder_bbf' in self.project:
                self.n_par_strms_tx = 4
                self.n_par_strms_rx = 4
            elif 'sounder_if' in self.project:
                self.n_par_strms_tx = 8
                self.n_par_strms_rx = 4
        else:
            if 'sounder_bbf' in self.project:
                self.n_par_strms_tx = 4
                self.n_par_strms_rx = 4
            elif 'sounder_if' in self.project:
                self.n_par_strms_tx = 4
                self.n_par_strms_rx = 4
        
        if 'ddr4' in self.project:
            if 'sounder_bbf' in self.project:
                self.tx_mode = 2
                self.rx_mode = 1
            elif 'sounder_if' in self.project:
                self.tx_mode = 1
                self.rx_mode = 1
        else:
            if 'sounder_bbf' in self.project:
                self.tx_mode = 1
                self.rx_mode = 1
            elif 'sounder_if' in self.project:
                self.tx_mode = 1
                self.rx_mode = 1

        self.n_skip = 0
        
        self.CLOCKWIZARD_LOCK_ADDRESS = 0x0004
        self.CLOCKWIZARD_RESET_ADDRESS = 0x0000
        self.CLOCKWIZARD_RESET_TOKEN = 0x000A

        self.txtd = None
        self.rxtd = None

        if self.RFFE=='sivers':
            self.init_sivers(params=params)
            # Now set beam index based on beamforming flag
            if self.params.beamforming:
                calculated_tx_index = self._calculate_beam_index_from_angles(self.params.steer_theta_deg, self.params.steer_phi_deg)
                # Assuming TX and RX use the same index for now
                calculated_rx_index = calculated_tx_index
                self.print(f"Beamforming enabled. Setting TX index to {calculated_tx_index}, RX index to {calculated_rx_index}", thr=1)
                success_tx, status_tx = self.siversControllerObj.setBeamIndexTX(calculated_tx_index)
                success_rx, status_rx = self.siversControllerObj.setBeamIndexRX(calculated_rx_index)
                if not success_tx or not success_rx:
                     self.print(f"Warning: Failed to set Sivers beam index. TX: {status_tx}, RX: {status_rx}", thr=0)
            else:
                # Default beam index if beamforming is disabled
                default_index = 32
                self.print(f"Beamforming disabled. Setting default TX/RX beam index to {default_index}", thr=1)
                self.siversControllerObj.setBeamIndexTX(default_index)
                self.siversControllerObj.setBeamIndexRX(default_index)
        elif self.RFFE=='piradio':
            pass
        elif self.RFFE=='none':
            pass

        self.load_bit_file()
        self.allocate_input(n_frame=self.n_frame_rd)
        self.allocate_output(n_frame=self.n_frame_wr)
        self.gpio_init()
        self.clock_init()
        self.verify_clock_tree()
        self.init_rfdc()
        if 'ddr4' in self.project:
            self.dac_tiles_sync_hex = 0x0
            for id in self.dac_tiles_sync:
                self.dac_tiles_sync_hex += 0x1 << id
            self.adc_tiles_sync_hex = 0x0
            for id in self.adc_tiles_sync:
                self.adc_tiles_sync_hex += 0x1 << id
            self.init_tile_sync()
            self.sync_tiles(dacTiles=self.dac_tiles_sync_hex, adcTiles=self.adc_tiles_sync_hex)
        self.init_dac()
        self.init_adc()
        if 'sounder_if' in self.project:
            self.set_dac_mixer()
            self.set_adc_mixer()
        self.dma_init()
        if self.run_tcp_server:
            self.tcp_comm = Tcp_Comm_RFSoC(params)
            self.tcp_comm.init_tcp_server()

        self.print("rfsoc initialization done", thr=1)


    def load_bit_file(self, verbose=False):
        self.print("Starting to load the bit-file", thr=1)

        self.ol = Overlay(self.bit_file_path)
        if verbose:
            self.ol.ip_dict
            # ol?

        self.print("Bit-file loading done", thr=1)


    def init_sivers(self, params=None):
        self.print("Starting Sivers EVK controller", thr=1)
        self.siversControllerObj = siversController(params)
        self.siversControllerObj.init()
        self.print("Sivers EVK controller is loaded", thr=1)


    def run_tcp(self):
        self.tcp_comm.obj_rfsoc = self
        self.tcp_comm.run_tcp_server(self.tcp_comm.parse_and_execute)
    

    def allocate_input(self, n_frame=1):
        size = self.n_rx_ant * n_frame * self.n_samples * 2
        if 'ddr4' in self.project:
            self.adc_rx_buffer = allocate(shape=(size,), target=self.ol.ddr4_0, dtype=np.int16)
            # self.adc_rx_buffer = allocate(shape=(size,), dtype=np.int16)
        else:
            self.adc_rx_buffer = allocate(shape=(size,), dtype=np.int16)
        self.print("Input buffers allocation done", thr=1)


    def allocate_output(self, n_frame=1):
        size = self.n_tx_ant * n_frame * self.n_samples * 2
        self.dac_tx_buffer = allocate(shape=(size,), dtype=np.int16)
        self.print("Output buffers allocation done", thr=1)


    def gpio_init(self):
        self.gpio_dic = {}

        if 'ddr4' in self.project:
            if self.board=='rfsoc_2x2':
                self.gpio_dic['lmk_reset'] = GPIO(GPIO.get_gpio_pin(84), 'out')
            elif self.board=='rfsoc_4x2':
                self.gpio_dic['lmk_reset'] = GPIO(GPIO.get_gpio_pin(-78+7), 'out')
            self.gpio_dic['dac_mux_sel'] = GPIO(GPIO.get_gpio_pin(3), 'out')
            self.gpio_dic['adc_enable'] = GPIO(GPIO.get_gpio_pin(34), 'out')
            self.gpio_dic['dac_enable'] = GPIO(GPIO.get_gpio_pin(2), 'out')
            self.gpio_dic['adc_reset'] = GPIO(GPIO.get_gpio_pin(32), 'out')
            self.gpio_dic['dac_reset'] = GPIO(GPIO.get_gpio_pin(0), 'out')
            self.gpio_dic['led'] = GPIO(GPIO.get_gpio_pin(80), 'out')
        else:
            self.gpio_dic['dac_mux_sel'] = GPIO(GPIO.get_gpio_pin(0), 'out')
            self.gpio_dic['adc_enable'] = GPIO(GPIO.get_gpio_pin(3), 'out')
            self.gpio_dic['dac_enable'] = GPIO(GPIO.get_gpio_pin(1), 'out')
            self.gpio_dic['adc_reset'] = GPIO(GPIO.get_gpio_pin(2), 'out')
            self.gpio_dic['dac_reset'] = GPIO(GPIO.get_gpio_pin(4), 'out')

        if 'ddr4' in self.project:
            self.gpio_dic['led'].write(0)
            self.gpio_dic['dac_mux_sel'].write(1)
            self.gpio_dic['adc_enable'].write(0)
            self.gpio_dic['dac_enable'].write(0)
            self.gpio_dic['adc_reset'].write(1)
            self.gpio_dic['dac_reset'].write(1)
        else:
            self.gpio_dic['dac_mux_sel'].write(0)
            self.gpio_dic['adc_enable'].write(0)
            self.gpio_dic['dac_enable'].write(0)
            self.gpio_dic['adc_reset'].write(0)
            self.gpio_dic['dac_reset'].write(0)

        self.print("PS-PL GPIOs initialization done", thr=1)


    def clock_init(self):
        if 'ddr4' in self.project:
            self.gpio_dic['lmk_reset'].write(1)
            self.gpio_dic['lmk_reset'].write(0)

        xrfclk.set_ref_clks(lmk_freq=self.lmk_freq_mhz, lmx_freq=self.lmx_freq_mhz)
        self.print("Xrfclk initialization done", thr=1)


    def verify_clock_tree(self):
        if 'ddr4' in self.project:
            status = self.ol.clocktreeMTS.clk_wiz_0.read(self.CLOCKWIZARD_LOCK_ADDRESS)
            if (status != 1):
                raise Exception("The MTS ClockTree has failed to LOCK. Please verify board clocking configuration")
        self.print("Verifying clock tree done", thr=1)


    def init_rfdc(self):
        self.rfdc = self.ol.usp_rf_data_converter_0
        self.print("RFDC initialization done", thr=1)


    def init_tile_sync(self):
        dacTiles = min(self.dac_tiles_sync_hex, 0x1)
        adcTiles = min(self.adc_tiles_sync_hex, 0x1)
        self.sync_tiles(dacTiles=dacTiles, adcTiles=adcTiles)
        self.ol.clocktreeMTS.clk_wiz_0.mmio.write_reg(self.CLOCKWIZARD_RESET_ADDRESS, self.CLOCKWIZARD_RESET_TOKEN)
        time.sleep(0.1)

        # for id in self.dac_tile_block_dic:
        for id in list(set(list(self.dac_tile_block_dic.keys()) + self.dac_tiles_sync)):
            self.rfdc.dac_tiles[id].Reset()

        for toggleValue in range(0,1):
            # for id in self.adc_tile_block_dic:
            for id in list(set(list(self.adc_tile_block_dic.keys()) + self.adc_tiles_sync)):
                self.rfdc.adc_tiles[id].SetupFIFO(toggleValue)
        self.print("Tiles sync initialization done", thr=1)
    

    def sync_tiles(self, dacTiles = 0, adcTiles = 0):
        self.rfdc.mts_dac_config.RefTile = 0  # MTS starts at DAC Tile 228
        self.rfdc.mts_adc_config.RefTile = 0  # MTS starts at ADC Tile 224
        self.rfdc.mts_dac_config.Target_Latency = -1
        self.rfdc.mts_adc_config.Target_Latency = -1
        if dacTiles > 0:
            self.rfdc.mts_dac_config.Tiles = dacTiles # group defined in binary 0b1111
            self.rfdc.mts_dac_config.SysRef_Enable = 1
            self.rfdc.mts_dac()
        else:
            self.rfdc.mts_dac_config.Tiles = 0x0
            self.rfdc.mts_dac_config.SysRef_Enable = 0

        if adcTiles > 0:
            self.rfdc.mts_adc_config.Tiles = adcTiles
            self.rfdc.mts_adc_config.SysRef_Enable = 1
            self.rfdc.mts_adc()
        else:
            self.rfdc.mts_adc_config.Tiles = 0x0
            self.rfdc.mts_adc_config.SysRef_Enable = 0
        self.print("Tiles sync done", thr=1)
    

    def init_dac(self):
        if 'sounder_if' in self.project and not 'ddr4' in self.project:
            # self.dac_tile.Reset()
            # self.dac_tile.SetupFIFO(True)

            for id in self.dac_tile_block_dic:
                self.rfdc.dac_tiles[id].Reset()
            for id in self.dac_tile_block_dic:
                self.rfdc.dac_tiles[id].SetupFIFO(True)
        self.print("DAC init and reset done", thr=1)


    def init_adc(self):
        if 'sounder_if' in self.project and not 'ddr4' in self.project:
            # # self.adc_tile.Reset()
            # # self.adc_tile.SetupFIFO(True)
            # for toggleValue in range(0, 1):
            #     self.adc_tile.SetupFIFO(toggleValue)

            # for id in self.adc_tile_block_dic:
            #     self.rfdc.adc_tiles[id].Reset()
            for toggleValue in range(0,1):
                for id in self.adc_tile_block_dic:
                    self.rfdc.adc_tiles[id].SetupFIFO(toggleValue)
        self.print("ADC init and reset done", thr=1)


    def set_dac_mixer(self):
        cofig_str = 'DAC configs: mix_freq: {:.2e}, mix_phase_off: {:.2f}'.format(self.mix_freq_dac, self.mix_phase_off)
        cofig_str += ', DynamicPLLConfig: ' + str(self.DynamicPLLConfig)
        cofig_str += ', do_mixer_settings: ' + str(self.do_mixer_settings)
        self.print(cofig_str, thr=2)

        for tile_id in self.dac_tile_block_dic:
            for block_id in self.dac_tile_block_dic[tile_id]:
                dac_tile = self.rfdc.dac_tiles[tile_id]
                dac_block = dac_tile.blocks[block_id]

                if self.do_pll_settings:
                    dac_tile.DynamicPLLConfig(self.DynamicPLLConfig[0], self.DynamicPLLConfig[1], self.DynamicPLLConfig[2])
                # print(dac_block.MixerSettings)
                if self.do_mixer_settings:
                    dac_block.MixerSettings['Freq'] = self.mix_freq_dac/1e6
                    dac_block.MixerSettings['PhaseOffset'] = self.mix_phase_off
                    # dac_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_IMMEDIATE
                    dac_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_TILE
                    dac_block.UpdateEvent(xrfdc.EVNT_SRC_TILE)

        self.print("DAC Mixer Settings done", thr=1)


    def set_adc_mixer(self):
        cofig_str = 'ADC configs: mix_freq: {:.2e}, mix_phase_off: {:.2f}'.format(self.mix_freq_adc, self.mix_phase_off)
        cofig_str += ', DynamicPLLConfig: ' + str(self.DynamicPLLConfig)
        cofig_str += ', do_mixer_settings: ' + str(self.do_mixer_settings)
        self.print(cofig_str, thr=2)

        for tile_id in self.adc_tile_block_dic:
            for block_id in self.adc_tile_block_dic[tile_id]:
                adc_tile = self.rfdc.adc_tiles[tile_id]
                adc_block = adc_tile.blocks[block_id]

                if self.do_pll_settings:
                    adc_tile.DynamicPLLConfig(self.DynamicPLLConfig[0], self.DynamicPLLConfig[1], self.DynamicPLLConfig[2])
                # print(adc_block.MixerSettings)
                # attributes = dir(adc_block.MixerSettings)
                # for name in attributes:
                #     print(name)
                if self.do_mixer_settings:
                    # adc_block.NyquistZone = 1
                    # adc_block.MixerSettings = {
                    #     'CoarseMixFreq'  : xrfdc.COARSE_MIX_BYPASS,
                    #     'EventSource'    : xrfdc.EVNT_SRC_TILE,
                    #     'FineMixerScale' : xrfdc.MIXER_SCALE_1P0,
                    #     'Freq'           : -1*mix_freq/1e6,
                    #     'MixerMode'      : xrfdc.MIXER_MODE_R2C,
                    #     'MixerType'      : xrfdc.MIXER_TYPE_FINE,
                    #     'PhaseOffset'    : 0.0
                    # }
                    adc_block.MixerSettings['Freq'] = -1*self.mix_freq_adc/1e6
                    adc_block.MixerSettings['PhaseOffset'] = self.mix_phase_off
                    # adc_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_IMMEDIATE
                    adc_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_TILE
                    # adc_block.UpdateEvent(xrfdc.EVENT_MIXER)
                    adc_block.UpdateEvent(xrfdc.EVNT_SRC_TILE)
                    # adc_block.MixerSettings['Freq'] = -1*self.mix_freq_adc/1e6
            
        self.print("ADC Mixer Settings done", thr=1)


    def dma_init(self):
        if 'ddr4' in self.project:
            self.ol.dac_path.axi_dma_0.set_up_tx_channel()
            self.dma_tx = self.ol.dac_path.axi_dma_0.sendchannel
        else:
            self.dma_tx = self.ol.TX_loop.axi_dma_tx.sendchannel
        self.print("TX DMA setup done", thr=1)

        if 'ddr4' in self.project:
            self.ol.adc_path.axi_dma_0.set_up_rx_channel()
            self.dma_rx = self.ol.adc_path.axi_dma_0.recvchannel
            self.rx_reg = self.ol.adc_path.axis_flow_ctrl_0
        else:
            self.dma_rx = self.ol.RX_Logic.axi_dma_rx.recvchannel
        self.print("RX DMA setup done", thr=1)


    def load_data_to_tx_buffer(self, txtd):
        self.txtd = txtd
        txtd_dac = self.txtd * (2 ** (self.dac_bits + 1) - 1)

        if 'sounder_if' in self.project:
            txtd_dac_interleaved = np.zeros(np.prod(txtd_dac.shape)*2, dtype='int16').reshape(-1, self.n_par_strms_tx//2, 2)
            for i in range(self.n_tx_ant):
                txtd_dac_ant = txtd_dac[i].reshape(-1,self.n_par_strms_tx//2)
                if self.tx_mode==1:
                    txtd_dac_interleaved[i::self.n_tx_ant,:,0] = np.int16(txtd_dac_ant.real)
                    txtd_dac_interleaved[i::self.n_tx_ant,:,1] = np.int16(txtd_dac_ant.imag)
                elif self.tx_mode==2:
                    txtd_dac_interleaved[i::self.n_tx_ant,:,0] = np.int16(txtd_dac_ant.imag)
                    txtd_dac_interleaved[i::self.n_tx_ant,:,1] = np.int16(txtd_dac_ant.real)
                else:
                    raise ValueError('Unsupported TX mode: %d' %(self.tx_mode))
            
        else:
            txtd_dac = txtd_dac.reshape(txtd_dac.shape[0], -1, self.n_par_strms_tx)
            txtd_dac_interleaved = np.zeros((np.prod(txtd_dac.shape)*2//self.n_par_strms_tx, self.n_par_strms_tx), dtype='int16')
            for i in range(self.n_tx_ant):
                if self.tx_mode==1:
                    txtd_dac_interleaved[i*2::self.n_tx_ant*2,:] = np.int16(txtd_dac[i].real)
                    txtd_dac_interleaved[i*2+1::self.n_tx_ant*2,:] = np.int16(txtd_dac[i].imag)
                elif self.tx_mode==2:
                    txtd_dac_interleaved[i*2::self.n_tx_ant*2,:] = np.int16(txtd_dac[i].imag)
                    txtd_dac_interleaved[i*2+1::self.n_tx_ant*2,:] = np.int16(txtd_dac[i].real)
                else:
                    raise ValueError('Unsupported RX mode: %d' %(self.rx_mode))

        self.dac_tx_buffer[:] = txtd_dac_interleaved.flatten()[:]

        self.print("Loading txtd data to DAC TX buffer done", thr=1)


    def load_data_from_rx_buffer(self):
        rx_data = np.array(self.adc_rx_buffer).astype('int16') / (2 ** (self.adc_bits + 1) - 1)
        self.rxtd = []
        
        rx_data = rx_data.reshape(-1, self.n_par_strms_rx)
        for i in range(self.n_rx_ant):
            if self.rx_mode==1:
                rx_data_ant = rx_data[i*2::self.n_rx_ant*2,:] + 1j * rx_data[i*2+1::self.n_rx_ant*2,:]
            elif self.rx_mode==2:
                rx_data_ant = rx_data[i*2+1::self.n_rx_ant*2,:] + 1j * rx_data[i*2::self.n_rx_ant*2,:]
            else:
                raise ValueError('Unsupported RX mode: %d' %(self.rx_mode))
            self.rxtd.append(rx_data_ant.flatten())
        
        self.rxtd = np.array(self.rxtd)
        self.print("Loading rxtd data from ADC RX buffer done", thr=5)


    def send_frame(self, txtd):
        self.load_data_to_tx_buffer(txtd)

        self.gpio_dic['dac_mux_sel'].write(0)
        self.gpio_dic['dac_enable'].write(0)
        if 'ddr4' in self.project:
            self.gpio_dic['dac_reset'].write(1)  # Reset ON
        else:
            self.gpio_dic['dac_reset'].write(0)
        time.sleep(0.5)
        if 'ddr4' in self.project:
            self.gpio_dic['dac_reset'].write(0)  # Reset OFF
        else:
            self.gpio_dic['dac_reset'].write(1)
        self.dma_tx.transfer(self.dac_tx_buffer)
        self.dma_tx.wait()
        self.gpio_dic['dac_mux_sel'].write(1)
        self.gpio_dic['dac_enable'].write(1)

        # self.dma_tx.wait()
        time.sleep(0.1)
        self.print("Frame sent via DAC", thr=1)


    def recv_frame_one(self, n_frame=1):
        # if 'ddr4' in self.project:
        #     self.gpio_dic['led'].write(1)
        if 'ddr4' in self.project:
            # Suspicous code
            self.rx_reg.write(0, self.n_rx_ant * self.n_samples // self.n_par_strms_rx)
            # self.rx_reg.write(0, self.n_samples // self.n_par_strms_rx)
            self.rx_reg.write(4, self.n_skip // self.n_par_strms_rx)
            self.rx_reg.write(8, self.n_rx_ant * n_frame * self.n_samples * 4)      # Must have self.n_rx_ant multiplier to work correctly

            self.gpio_dic['adc_reset'].write(0)
        else:
            self.gpio_dic['adc_enable'].write(0)
            self.gpio_dic['adc_reset'].write(0)  # Reset ON
            time.sleep(0.01)
            self.gpio_dic['adc_reset'].write(1)  # Reset OFF

        self.dma_rx.transfer(self.adc_rx_buffer)
        self.gpio_dic['adc_enable'].write(1)
        self.dma_rx.wait()

        self.gpio_dic['adc_enable'].write(0)

        if 'ddr4' in self.project:
            self.gpio_dic['adc_reset'].write(1)
        else:
            self.gpio_dic['adc_reset'].write(0)

        self.load_data_from_rx_buffer()
        # if 'ddr4' in self.project:
        #     self.gpio_dic['led'].write(0)
        self.print("Frames received from ADC", thr=5)

        return self.rxtd
    

    def recv_frame(self, n_frame=1):
        beam_indices_to_use = []
        num_beams_to_process = 0

        if self.RFFE == 'sivers':
            if self.params.beamforming:
                # Get the single beam index set during __init__
                # No need to call getBeamIndexRX here if we trust __init__ set it.
                # We just need a list containing the single index *conceptually*.
                # For the loop, we can just use a dummy list of size 1, or get the actual index. Let's get it.
                try:
                    current_rx_index = self.siversControllerObj.getBeamIndexRX()
                    beam_indices_to_use = [current_rx_index]
                    num_beams_to_process = 1
                    self.print(f"Receiving frame with beamforming enabled (using pre-set RX index: {current_rx_index})", thr=2)
                except Exception as e:
                    self.print(f"Error getting beam index, defaulting to beam_test: {e}", thr=0)
                    beam_indices_to_use = self.beam_test
                    num_beams_to_process = len(beam_indices_to_use)
            else:
                # Use the test list if beamforming is off
                beam_indices_to_use = self.beam_test
                num_beams_to_process = len(beam_indices_to_use)
                self.print(f"Receiving frames by sweeping through beam_test: {beam_indices_to_use}", thr=2)
        else:
            # Non-Sivers case: treat as a single receive operation
            beam_indices_to_use = [0] # Dummy index for the loop structure
            num_beams_to_process = 1
            self.print("Receiving frame (non-Sivers or specific case)", thr=2)

        # Allocate space based on the number of beams we will actually process
        rxtd = np.zeros((num_beams_to_process, self.n_rx_ant * n_frame * self.n_samples), dtype='complex')

        for i, beam_index in enumerate(beam_indices_to_use):
            # Only set beam index if Sivers RFFE and beamforming is OFF (sweeping through list)
            if self.RFFE == 'sivers' and not self.params.beamforming:
                success, status = self.siversControllerObj.setBeamIndexRX(beam_index)
                if not success:
                    self.print(f"Warning: Failed to set RX beam index {beam_index} during receive loop: {status}", thr=0)
                    # Decide how to handle failure: continue, skip, raise error? For now, just print warning.

            # Perform the actual receive for this configuration/beam
            # recv_frame_one updates self.rxtd internally
            self.recv_frame_one(n_frame=n_frame)

            # Store the received data from self.rxtd
            # Ensure flatten() handles the shape returned by recv_frame_one correctly.
            # self.rxtd should have shape (n_rx_ant, n_samples*n_frame)
            if self.rxtd is not None:
                 rxtd[i, :] = self.rxtd.flatten()
            else:
                 self.print(f"Warning: self.rxtd is None after recv_frame_one for beam index {beam_index}", thr=0)
                 # Handle error case, maybe fill with zeros or skip? Fill with zeros for now.
                 rxtd[i, :] = 0


        # --- Common post-processing remains the same ---
        # This processing naturally handles rxtd having 1 or multiple rows (beams)
        rxfd = fft(rxtd, axis=1)
        rxfd = np.roll(rxfd, 1, axis=1)

        n_samples_rx_axis = rxfd.shape[-1]
        # Check if self.txtd exists and has data before using it
        # Also ensure txfd matches the required dimension length
        hest = None # Initialize hest
        if self.txtd is not None and self.txtd.size > 0 and self.txtd.shape[-1] >= n_samples_rx_axis:
            txfd_conj = np.conj(self.txtd[:, :n_samples_rx_axis])
            # Ensure txfd_conj can broadcast with rxfd (e.g., if txfd is (1, N) and rxfd is (M, N))
            if txfd_conj.shape[0] == 1 and rxfd.shape[0] > 1:
                 txfd_conj = np.tile(txfd_conj, (rxfd.shape[0], 1)) # Tile if necessary
            elif txfd_conj.shape[0] != rxfd.shape[0] and rxfd.shape[0] != 1:
                 # Handle potential shape mismatch if not simple broadcasting
                 self.print(f"Warning: Shape mismatch for Hest calculation. rxfd: {rxfd.shape}, txfd_conj: {txfd_conj.shape}", thr=1)
                 # Defaulting to raw rxfd as estimation failed. Adjust as needed.
                 hest = rxfd
            
            if hest is None: # If no shape mismatch or handled
                 Hest = rxfd * txfd_conj
                 hest = ifft(Hest, axis=1)
        else:
            # Handle case where txtd is not available or has incorrect shape
            warning_msg = "Warning: self.txtd not available"
            if self.txtd is not None:
                warning_msg += f" or shape mismatch (txtd: {self.txtd.shape[-1]}, needed: {n_samples_rx_axis})"
            warning_msg += " for channel estimation. Returning raw rxfd."
            self.print(warning_msg, thr=1)
            hest = rxfd # Or return None, or raise an error, depending on desired behavior


        self.print("Frames received from ADC and processed", thr=5)
        return hest # Return estimated channel(s) or raw RX FD data if estimation failed

    # Placeholder function to calculate beam index from angles
    # TODO: Replace this with the actual Sivers beam index calculation logic
    # based on your antenna array and Sivers beam codebook.
    def _calculate_beam_index_from_angles(self, theta_deg, phi_deg):
        # Example placeholder: Linearly map phi angle (-90 to +90) to index (0-63)
        # This is likely incorrect for the actual hardware.
        index = int(np.clip((phi_deg + 90.0) / 180.0 * 63.0, 0, 63))
        print(f"Placeholder: Calculated beam index {index} for theta={theta_deg}, phi={phi_deg}", thr=1) # Use self.print if inside class
        return index


