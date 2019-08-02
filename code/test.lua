require 'nn'
require 'nngraph' 
require 'qtcore'
require 'qtgui'

app = QApplication.new(1, {'lua'}, 1)

      -- Process inputs
        local model_input = nn.Sequential()
        model_input:add(nn.CAddTable(2))

        local model_state = nn.Sequential()
        model_state:add(nn.LookupTable(2, 1000))

        -- Output
        local model_out = nn.Sequential()
        model_out:add(nn.Linear(100, 100))
        model_out:add(nn.ReLU(true))
        model_out:add(nn.Linear(100, 8))

        -- Construct Graph
        local in_state = nn.Identity()()
        local in_id = nn.Identity()()


        local in_all = {
	    model_state(in_state),
            nn.LookupTable(2, 100)(in_id)
        }


        -- Process inputs
        local proc_input = model_input(in_all)

        -- Process out
        local proc_out = model_out(proc_input)

        -- Create model
        local model_inputs = { in_state, in_id}
        local model_outputs = { proc_out }


        nngraph.annotateNodes()

        local model = nn.gModule(model_inputs, model_outputs)

  graph.dot(model.fg, 'MLP')

app.exec()
