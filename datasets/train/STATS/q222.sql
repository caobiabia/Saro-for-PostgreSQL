select  count(*) from comments as c,  		postHistory as ph,          votes as v,  		users as u where u.Id  = v.UserId  	and v.UserId = ph.UserId  	and ph.UserId =c.UserId  AND c.Score=0  AND u.Reputation>=1  AND u.Views>=0  AND u.Views<=211  AND v.CreationDate>='2009-02-03 00:00:00'::timestamp;