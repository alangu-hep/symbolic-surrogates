import torch

class HookHandler():
    def __init__(self, model: torch.nn.Module, hook_manager: dict):
        
        self.model = model if not hasattr(model, 'mod') else model.mod
        self.manager = hook_manager
        
        self.handles = {
            hook_type: {name: value for name, value in hook_dict.items()} 
            for hook_type, hook_dict in self.manager.items()}
        self.hook_outputs = {
            hook_type: {name: [] for name in hook_dict.keys()} 
            for hook_type, hook_dict in self.manager.items()}

    @staticmethod
    def clear_handles(handles: dict):
        print('\nRemoving all handles')

        for hook_type, hook_dict in handles.items():
            for handle in hook_dict.values():
                handle.remove()

    @staticmethod
    def remove_all_hooks(model: torch.nn.Module):
        
        print('\nRemoving all hooks within the model')
        
        for _, module in model.named_modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            
    @staticmethod
    def save_inputs(name, outputs):
        def hook(module, input):
        
            location = outputs[name]
            
            if isinstance(input, torch.Tensor):
                location.append(input.detach())
                return
            
            if isinstance(input, (tuple, list)):
                location.append([
                    i.detach() if isinstance(i, torch.Tensor) else i
                    for i in input
                ])
                return
                
            location.append(input)
        
        return hook

    @staticmethod
    def save_outputs(name, outputs):
        def hook(module, input, output):
        
            location = outputs[name]
            
            if isinstance(output, torch.Tensor):
                location.append(output.detach())
                return
            
            if isinstance(output, (tuple, list)):
                location.append([
                    o.detach() if isinstance(o, torch.Tensor) else o
                    for o in output
                ])
                return
                
            location.append(output)
        
        return hook

    def registration(self, safety_remove = False):
        if safety_remove:
            HookHandler.remove_all_hooks(self.model)
        
        for hook_type, hook_dict in self.manager.items():  
            try:
                for hook_name, attr in hook_dict.items():
                    
                    attr_object = getattr(self.model, attr)
                    
                    if hook_type == 'forward_hooks':
                        self.handles[hook_type][hook_name] = attr_object.register_forward_hook(HookHandler.save_outputs(hook_name, self.hook_outputs[hook_type]))
                        print(f'\nForward Hook Registered: {hook_name}')
                    elif hook_type == 'pre_forward_hooks':
                        self.handles[hook_type][hook_name] = attr_object.register_forward_pre_hook(HookHandler.save_inputs(hook_name, self.hook_outputs[hook_type]))
                        print(f'\nForward Hook Registered: {hook_name}')
                        
            except TypeError:
                print('\nFailed to register hooks. Resetting model')
                HookHandler.remove_all_hooks(self.model)
                break
        
        return self.handles, self.hook_outputs