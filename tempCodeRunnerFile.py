ult = torch.norm(lora_B, dim = 1) ** 2
            # result = lora_B @ lora_A
            result, _ = torch.sort(result.view(-1).abs(), descending=True)
            x = range(1, result.numel()+1)
            plt.plot(x, result, label='Iteration:'+str(i))

            #  拟合幂律分布
            def func(x, alpha, c):
                # return c * np.log(alpha*x)
                # return c * np.exp(-alpha*x)
                return c * np.power(x, alpha)
            
            x = torch.tensor(range(1, result.numel() + 1))
            popt, pcov = curve_fit(func, x.numpy(), result.numpy())
            # plt.plot(x, popt[1]*np.log(x*popt[0]), label='Iteration:'+str(i), linestyle='--')
            # plt.plot(x, popt[1]*np.exp(-x*popt[0]), label='Iteration:'+str(i), linestyle='--')
            plt.plot(x, popt[1]*np.power(x, popt[0]), label='Iteration:'+str(i), linestyle='--')
            bre