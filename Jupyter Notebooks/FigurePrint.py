

plt.figure(1)
plt.axes().set_aspect(1)
plt.grid(True)
plt.title('Flat-Fading Channel Constellation', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('REAL', fontsize=10)
plt.ylabel('IMAG', fontsize=10)
plt.scatter(tf.math.real(x), tf.math.imag(x), s=16, c='b', label='TX')
plt.scatter(tf.math.real(x_hat_zf), tf.math.imag(x_hat_zf), s=16, c='y', label='ZF')
plt.scatter(tf.math.real(x_hat_lmmse), tf.math.imag(x_hat_lmmse), s=16, c='g', label='LMMSE')
plt.scatter(tf.math.real(x_hat_dip), tf.math.imag(x_hat_dip), s=16, c='r', label='DIP')
plt.legend(loc='lower left', fontsize=8)
plt.tight_layout()

plt.figure(2)
title = "SER: Noncoding MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
xlabel = "$E_b/N_0$ (dB)"
ylabel = "SER (log)"
plt.title(title, fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(xlabel, fontsize=10)
plt.ylabel(ylabel, fontsize=10)
plt.grid(which="both")
plt.semilogy(snrs, sers_zf_mean, 'b', label='ZF')
plt.semilogy(snrs, sers_lmmse_mean, 'g', label='LMMSE')
plt.semilogy(snrs, sers_dip_mean, 'r', label='DIP')
plt.legend(loc='lower left', fontsize=8)
plt.tight_layout()

plt.show()